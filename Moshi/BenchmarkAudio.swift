import AVFoundation
import Foundation

/// Bounded microphone queue. Inference runs on a worker, never on the audio tap.
final class AudioInput: @unchecked Sendable {
    struct Packet { let pcm: [Float]; let capturedAt: Double }
    private let engine = AVAudioEngine()
    private let condition = NSCondition()
    private var packets: [Packet] = []
    private var stopped = false
    private var dropped = 0
    private var pending: [Float] = []
    private var pendingAt: Double = 0
    private var installed = false
    var hardwareSampleRate: Double = 0
    static let sampleRate = 24000.0
    static let frameSamples = 1920

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setPreferredSampleRate(Self.sampleRate)
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true)
        let input = engine.inputNode
        let sourceFormat = input.outputFormat(forBus: 0)
        hardwareSampleRate = sourceFormat.sampleRate
        guard sourceFormat.sampleRate > 0,
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Self.sampleRate, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: sourceFormat, to: format) else {
            throw CustomError("Microphone format unavailable")
        }
        input.installTap(onBus: 0, bufferSize: 1920, format: sourceFormat) { [weak self] buffer, _ in
            guard let self else { return }
            let timestamp = ProcessInfo.processInfo.systemUptime - Double(buffer.frameLength) / sourceFormat.sampleRate
            let capacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * Self.sampleRate / sourceFormat.sampleRate)) + 64
            guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return }
            var supplied = false
            var error: NSError?
            converter.convert(to: output, error: &error) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true
                status.pointee = .haveData
                return buffer
            }
            guard error == nil, let samples = output.floatChannelData else { return }
            self.condition.lock()
            defer { self.condition.unlock() }
            guard !self.stopped else { return }
            if self.pending.isEmpty { self.pendingAt = timestamp }
            self.pending.append(contentsOf: UnsafeBufferPointer(start: samples[0], count: Int(output.frameLength)))
            while self.pending.count >= Self.frameSamples {
                let packet = Packet(pcm: Array(self.pending.prefix(Self.frameSamples)), capturedAt: self.pendingAt)
                self.pending.removeFirst(Self.frameSamples)
                self.pendingAt += 0.08
                if self.packets.count == 8 { self.packets.removeFirst(); self.dropped += 1 }
                self.packets.append(packet)
                self.condition.signal()
            }
        }
        installed = true
        engine.prepare()
        try engine.start()
    }
    func receive() -> Packet? {
        condition.lock(); defer { condition.unlock() }
        if packets.isEmpty && !stopped { _ = condition.wait(until: Date().addingTimeInterval(0.1)) }
        return packets.isEmpty ? nil : packets.removeFirst()
    }
    func counters() -> (dropped: Int, queuedMs: Double) {
        condition.lock(); defer { condition.unlock() }
        return (dropped, Double(packets.count) * 80)
    }
    func stop() {
        condition.lock(); stopped = true; condition.broadcast(); condition.unlock()
        engine.stop()
        if installed { engine.inputNode.removeTap(onBus: 0); installed = false }
    }
}

/// Fixed-capacity output ring. Missing samples are counted only after playback begins.
final class AudioOutput: @unchecked Sendable {
    static let prefillSamples = 9600 // Five frames: absorb the F2 160 ms transient plus one frame margin.
    private let engine = AVAudioEngine()
    private let lock = NSLock()
    private var ring = [Float](repeating: 0, count: 24000)
    private var read = 0, write = 0, count = 0
    private var started = false
    private var underruns = 0, missingSamples = 0, rejectedFrames = 0
    private var firstRender: Double?
    func start() throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!
        let source = AVAudioSourceNode(format: format) { [weak self] _, _, frameCount, list in
            guard let self else { return noErr }
            self.lock.lock(); defer { self.lock.unlock() }
            let buffers = UnsafeMutableAudioBufferListPointer(list)
            let wanted = Int(frameCount)
            let available = self.started ? min(wanted, self.count) : 0
            if available > 0 && self.firstRender == nil { self.firstRender = ProcessInfo.processInfo.systemUptime }
            if self.started && available < wanted { self.underruns += 1; self.missingSamples += wanted - available }
            for i in 0..<wanted {
                let value: Float
                if i < available { value = self.ring[self.read]; self.read = (self.read + 1) % self.ring.count }
                else { value = 0 }
                for buffer in buffers { buffer.mData?.assumingMemoryBound(to: Float.self)[i] = value }
            }
            self.count -= available
            return noErr
        }
        engine.attach(source)
        engine.connect(source, to: engine.mainMixerNode, format: format)
        try engine.start()
    }
    func send(_ values: [Float]) {
        lock.lock(); defer { lock.unlock() }
        guard count + values.count <= ring.count else { rejectedFrames += 1; return }
        for value in values { ring[write] = value; write = (write + 1) % ring.count }
        count += values.count
        if count >= Self.prefillSamples { started = true }
    }
    func counters() -> (underruns: Int, missing: Int, rejected: Int, queuedMs: Double, firstRender: Double?) {
        lock.lock(); defer { lock.unlock() }
        return (underruns, missingSamples, rejectedFrames, Double(count) / 24, firstRender)
    }
    func stop() { engine.stop() }
}
