import SwiftUI
import AVFoundation
import MLX
import MoshiLib
import Synchronization

@MainActor @Observable
final class BenchmarkController {
    var configuration = BenchmarkConfiguration()
    var running = false
    var status = "Ready"
    var latest = BenchmarkMetrics()
    var logURL: URL?
    private var worker: Task<Void, Never>?
    private let stopFlag = Atomic<Bool>(false)
    private let stopCause = Mutex<String?>(nil)
    func stop(reason: String = "user") {
        stopCause.withLock { $0 = reason }
        stopFlag.store(true, ordering: .relaxed); status = "Stopping…"
    }
    func start() {
        guard !running else { return }
        running = true; status = "Loading"; logURL = nil; latest = BenchmarkMetrics()
        stopFlag.store(false, ordering: .relaxed)
        stopCause.withLock { $0 = nil }
        UIApplication.shared.isIdleTimerDisabled = true
        let config = configuration
        UIDevice.current.isBatteryMonitoringEnabled = true
        worker = Task.detached(priority: .userInitiated) { [self] in
            do { try await self.run(config) }
            catch { await self.finished("Failed: \(error)") }
        }
    }
    private func finished(_ message: String) {
        status = message; running = false
        UIApplication.shared.isIdleTimerDisabled = false
        print("BENCHMARK: \(message)")
    }
    private func update(_ sample: BenchmarkMetrics) { latest = sample; status = "Running" }

    nonisolated private func run(_ config: BenchmarkConfiguration) async throws {
        FrameDiagnostics.detailedCodec = ProcessInfo.processInfo.arguments.contains("--profile-codec")
        GPU.set(cacheLimit: 64 * 1024 * 1024)
        GPU.clearCache()
        GPU.resetPeakMemory()
        let baseline = residentMemory()
        let loader = BenchmarkModelLoader()
        let loadStart = ProcessInfo.processInfo.systemUptime
        var transformer: TransformerRunner?
        if config.mode != "mimi" {
            transformer = try await TransformerRunner(selection: config.modelSelection, contextLimit: config.contextSteps, loader: loader, optimizedGeneration: config.optimizedGeneration, halfPrecisionActivations: config.halfPrecisionActivations, temporalWindow: config.temporalWindow)
        }
        let afterModel = residentMemory()
        var mimi: Mimi?
        if config.mode != "transformer" { mimi = try await loader.makeMimi(numCodebooks: transformer?.includesDepthTransformer == true ? 8 : 16, halfPrecision: config.halfPrecisionCodec); eval(mimi!) }
        let afterCodec = residentMemory()
        let loadSeconds = ProcessInfo.processInfo.systemUptime - loadStart
        if config.verifyGeneration { try transformer?.verifyGenerationParity() }
        var codecValidation: [String: Double] = [:]
        if config.verifyGeneration && config.halfPrecisionCodec, let mimi {
            codecValidation = try await loader.verifyCodecPrecision(mimi)
            if config.contiguousCodecWeights {
                codecValidation.merge(try await loader.verifyCodecExecution(mimi)) { _, new in new }
            }
        }
        let warmupStarted = ProcessInfo.processInfo.systemUptime
        // Compile the exact streaming shapes before opening the microphone queue.
        // Record this cold cost separately, then reset all stream/cache state.
        if let mimi {
            let codes = mimi.encodeStep(StreamArray(MLXArray.zeros([1, 1, 1920])))
            codes.eval()
            if let array = codes.asArray() { mimi.decodeStep(StreamArray(array)).eval() }
            mimi.resetState()
        }
        for _ in 0..<3 {
            if let generated = transformer?.step(codes: nil), let mimi {
                mimi.decodeStep(StreamArray(generated)).eval()
            }
        }
        mimi?.resetState()
        transformer?.reset()
        let coldWarmupSeconds = ProcessInfo.processInfo.systemUptime - warmupStarted
        // Fixed, nonzero, changing 1-second PCM fixture for unpaced full-pipeline comparisons.
        // Same 24 kHz input on both paths; no microphone wait or speaker-rate cap.
        let fixture: [Float] = (0..<24000).map { index in
            let t = Double(index) / 24000
            return Float(0.08 * sin(2 * .pi * 173 * t) + 0.03 * sin(2 * .pi * 431 * t))
        }
        let startingThermalState = thermalName()
        let startingBattery = await MainActor.run { (UIDevice.current.batteryLevel, UIDevice.current.batteryState.rawValue) }
        let startingBatteryState = [0: "unknown", 1: "unplugged", 2: "charging", 3: "full"][startingBattery.1] ?? "unknown"
        print("BENCHMARK preflight: thermal=\(startingThermalState) battery=\(startingBatteryState)")
        if config.temporalWindow != nil && ProcessInfo.processInfo.arguments.contains("--acceptance") && startingThermalState != "nominal" {
            throw CustomError("Bounded-window acceptance requires nominal thermal start")
        }
        if ProcessInfo.processInfo.arguments.contains("--acceptance") {
            guard startingBatteryState == "unplugged",
                  startingThermalState == "nominal" || startingThermalState == "fair" else {
                throw CustomError("Acceptance preflight rejected: thermal=\(startingThermalState), battery=\(startingBatteryState)")
            }
        }
        let input = AudioInput(), output = AudioOutput()
        defer { input.stop(); output.stop() }
        if config.mode == "combined" || config.mode == "mimi" {
            let granted = await AVCaptureDevice.requestAccess(for: .audio)
            guard granted else { throw CustomError("Microphone permission is required") }
            try input.start(); try output.start()
        }
        let logger = try BenchmarkLogger()
        await MainActor.run { self.logURL = logger.url }
        let batteryStart = await MainActor.run { UIDevice.current.batteryLevel }
        try logger.write(BenchmarkHeader(runID: logger.url.deletingPathExtension().lastPathComponent,
            startedAt: Date(), device: deviceIdentifier(), ios: ProcessInfo.processInfo.operatingSystemVersionString,
            configuration: config, loadSeconds: loadSeconds, coldWarmupSeconds: coldWarmupSeconds, playbackPrefillSamples: AudioOutput.prefillSamples, modelFileBytes: transformer?.sourceFileBytes ?? 0,
            modelSource: transformer?.sourceName ?? "Mimi tokenizer-dbaa9758-checkpoint125.safetensors",
            precision: transformer?.precision ?? "Mimi checkpoint native dtype", logicalParameters: transformer?.parameterCount ?? 0,
            includesDepthTransformer: transformer?.includesDepthTransformer ?? false,
            parameterBreakdown: transformer?.parameterBreakdown ?? [:],
            audioTokenSource: transformer?.includesDepthTransformer == true ? "generated-depth-transformer" : "microphone-forwarded",
            architecture: config.modelSelection == "complete-random-q4" ? "temporal:14x2048,16heads,ff8192; depth:8slices x6layers x512,8heads,ff2048; text:8000; audio:2x8x2048; delay:2; seed:42; Q4/group64; pretrained Mimi8" : "upstream selected configuration",
            inputSource: config.mode == "throughput" ? "unpaced deterministic 24kHz two-tone PCM fixture; output evaluated and copied, no playback" : config.mode == "transformer" ? "zero audio tokens" : "live microphone",
            generationValidation: transformer?.generationValidation ?? "not applicable",
            codecValidation: codecValidation,
            modelWeightTensorBytes: transformer?.model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes } ?? 0,
            mimiWeightTensorBytes: mimi?.benchmarkWeightBytes ?? [:], mlxCacheLimitBytes: GPU.cacheLimit,
            baselineResidentBytes: baseline, afterModelResidentBytes: afterModel, afterCodecResidentBytes: afterCodec,
            hardwareSampleRate: input.hardwareSampleRate, batteryStart: batteryStart,
            startingThermalState: startingThermalState, startingBatteryState: startingBatteryState))
        let started = ProcessInfo.processInfo.systemUptime
        var lastLog = started, lastCPU = processCPUSeconds(), lastSteps = 0
        var sample = BenchmarkMetrics()
        var encodeTotal = 0.0, decodeTotal = 0.0, transformerTotal = 0.0, frameCount = 0
        var temporalTotal = 0.0, depthTotal = 0.0
        var memorySum = 0.0, memorySamples = 0
        var frameTimes: [Double] = []
        var traces: [FrameTrace] = []; traces.reserveCapacity(config.durationSeconds * 13)
        var previousEnd = started
        var firstCapture: Double?
        while !stopFlag.load(ordering: .relaxed) {
            let now = ProcessInfo.processInfo.systemUptime
            if now - started >= Double(config.durationSeconds) { break }
            var packet: AudioInput.Packet?
            if config.mode == "throughput" {
                let offset = (sample.steps * 1920) % fixture.count
                let pcm = (0..<1920).map { fixture[(offset + $0) % fixture.count] }
                packet = AudioInput.Packet(pcm: pcm, capturedAt: now)
            } else if config.mode != "transformer" {
                packet = input.receive()
                if packet == nil {
                    if now - lastLog > 5 { throw CustomError("No microphone frames for five seconds") }
                    continue
                }
                if firstCapture == nil { firstCapture = packet!.capturedAt }
            }
            let computeStarted = ProcessInfo.processInfo.systemUptime
            FrameDiagnostics.enabled = true
            FrameDiagnostics.reset()
            autoreleasepool {
                var codes: MLXArray?
                if let packet, let mimi {
                    let t = ProcessInfo.processInfo.systemUptime
                    let stream = FrameDiagnostics.measure("encodeGraph") { mimi.encodeStep(StreamArray(MLXArray(packet.pcm).reshaped([1, 1, 1920]))) }
                    FrameDiagnostics.measure("encodeEvalWait") { stream.eval() }; codes = stream.asArray()
                    encodeTotal += (ProcessInfo.processInfo.systemUptime - t) * 1000
                }
                var generated: MLXArray?
                if let transformer {
                    if sample.contextPosition >= config.contextSteps {
                        transformer.reset(); sample.contextPosition = 0; sample.contextResets += 1
                    }
                    let t = ProcessInfo.processInfo.systemUptime
                    generated = transformer.step(codes: codes)
                    temporalTotal += transformer.temporalMs
                    depthTotal += transformer.depthMs
                    if let generated {
                        let tokens = generated.asArray(Int32.self)
                        sample.generatedAudioFrames += 1
                        sample.generatedAudioTokens += tokens.count
                        sample.generatedTokenChecksum += tokens.reduce(Int64(0)) { $0 + Int64($1) }
                        sample.invalidGeneratedTokens += tokens.filter { $0 < 0 || $0 >= 2048 }.count
                    }
                    transformerTotal += (ProcessInfo.processInfo.systemUptime - t) * 1000
                    sample.contextPosition += 1
                }
                // A complete model decodes ONLY its generated codebooks, never input fallback.
                let outputCodes = transformer?.includesDepthTransformer == true ? generated : codes
                if let codes = outputCodes, let mimi {
                    let t = ProcessInfo.processInfo.systemUptime
                    let decoded = FrameDiagnostics.measure("decodeGraph") { mimi.decodeStep(StreamArray(codes)) }
                    FrameDiagnostics.measure("decodeEvalWait") { decoded.eval() }
                    if let pcm = decoded.asArray() {
                        let values = pcm.asArray(Float.self)
                        sample.decodedSamples += values.count
                        sample.nonfiniteDecodedSamples += values.filter { !$0.isFinite }.count
                        // Random tokens can make harsh noise; attenuate playback only, outside inference.
                        let playback = transformer?.includesDepthTransformer == true
                            ? values.map { $0.isFinite ? max(-1, min(1, $0)) * 0.1 : 0 } : values
                        if config.mode != "throughput" { FrameDiagnostics.measure("audioSend") { output.send(playback) } }
                    }
                    decodeTotal += (ProcessInfo.processInfo.systemUptime - t) * 1000
                }
                sample.steps += 1; frameCount += 1
                if let packet { sample.captureToEnqueueMs = (ProcessInfo.processInfo.systemUptime - packet.capturedAt) * 1000 }
            }
            // Explicit trailing fence tests for work deferred beyond evaluated PCM.
            FrameDiagnostics.measure("trailingMetalSync") { Stream.gpu.synchronize() }
            let end = ProcessInfo.processInfo.systemUptime
            let frameMs = (end - computeStarted) * 1000
            frameTimes.append(frameMs)
            sample.maxFrameComputeMs = max(sample.maxFrameComputeMs, frameMs)
            if frameMs > 80 { sample.computeDeadlineMisses += 1 }
            let allocation = GPU.snapshot(), incomingFrame = input.counters(), outgoingFrame = output.counters()
            traces.append(FrameTrace(step: sample.steps, elapsedSeconds: end - started, computeMs: frameMs,
                workerGapMs: (now - previousEnd) * 1000, receiveWaitMs: (computeStarted - now) * 1000,
                timings: FrameDiagnostics.milliseconds, operations: FrameDiagnostics.counts,
                mlxActiveBytes: allocation.activeMemory, mlxCacheBytes: allocation.cacheMemory, mlxPeakBytes: allocation.peakMemory,
                inputQueuedMs: incomingFrame.queuedMs, outputQueuedMs: outgoingFrame.queuedMs, underruns: outgoingFrame.underruns))
            previousEnd = end
            FrameDiagnostics.enabled = false
            if end - lastLog >= 1 {
                let sortedTimes = frameTimes.sorted()
                sample.frameComputeP50Ms = sortedTimes[sortedTimes.count / 2]
                sample.frameComputeP95Ms = sortedTimes[min(sortedTimes.count - 1, Int(Double(sortedTimes.count) * 0.95))]
                frameTimes.removeAll(keepingCapacity: true)
                sample.elapsedSeconds = end - started
                sample.stepsPerSecond = Double(sample.steps - lastSteps) / (end - lastLog)
                sample.encodeMs = encodeTotal / Double(frameCount)
                sample.transformerMs = transformerTotal / Double(frameCount)
                sample.temporalMs = temporalTotal / Double(frameCount)
                sample.depthMs = depthTotal / Double(frameCount)
                sample.decodeMs = decodeTotal / Double(frameCount)
                sample.computeRealtimeFactor = (sample.encodeMs + sample.transformerMs + sample.decodeMs) / 80
                sample.transformerKVBytes = transformer?.model.benchmarkKVCacheBytes ?? 0
                sample.mimiKVBytes = mimi?.benchmarkKVCacheBytes ?? 0
                sample.physicalFootprintBytes = physicalFootprint()
                sample.residentBytes = residentMemory()
                sample.peakSampledResidentBytes = max(sample.peakSampledResidentBytes, sample.residentBytes)
                memorySum += Double(sample.residentBytes); memorySamples += 1
                sample.averageSampledResidentBytes = memorySum / Double(memorySamples)
                let gpu = GPU.snapshot()
                sample.mlxActiveBytes = gpu.activeMemory; sample.mlxCacheBytes = gpu.cacheMemory; sample.mlxPeakBytes = gpu.peakMemory
                let cpu = processCPUSeconds()
                sample.cpuPercent = 100 * (cpu - lastCPU) / (end - lastLog); lastCPU = cpu
                sample.thermalState = thermalName()
                let battery = await MainActor.run { (UIDevice.current.batteryLevel, UIDevice.current.batteryState.rawValue) }
                sample.batteryLevel = battery.0
                sample.batteryState = [0: "unknown", 1: "unplugged", 2: "charging", 3: "full"][battery.1] ?? "unknown"
                let incoming = input.counters(), outgoing = output.counters()
                sample.droppedInputFrames = incoming.dropped; sample.inputQueuedMs = incoming.queuedMs
                sample.underrunCallbacks = outgoing.underruns; sample.missingOutputSamples = outgoing.missing
                sample.rejectedOutputFrames = outgoing.rejected; sample.outputQueuedMs = outgoing.queuedMs
                if let firstCapture, let firstRender = outgoing.firstRender { sample.firstCaptureToRenderMs = (firstRender - firstCapture) * 1000 }
                try logger.write(sample)
                print("BENCHMARK sample: \(sample.elapsedSeconds)s \(sample.stepsPerSecond) steps/s thermal=\(sample.thermalState)")
                await update(sample)
                lastLog = end; lastSteps = sample.steps
                temporalTotal = 0; depthTotal = 0
                encodeTotal = 0; decodeTotal = 0; transformerTotal = 0; frameCount = 0
            }
        }
        let completedSeconds = ProcessInfo.processInfo.systemUptime - started
        input.stop(); output.stop()
        // Capture final counters, closing the F2 last-fractional-second evidence gap.
        let finalInput = input.counters(), finalOutput = output.counters()
        sample.elapsedSeconds = completedSeconds
        sample.droppedInputFrames = finalInput.dropped; sample.inputQueuedMs = finalInput.queuedMs
        sample.underrunCallbacks = finalOutput.underruns; sample.missingOutputSamples = finalOutput.missing
        sample.rejectedOutputFrames = finalOutput.rejected; sample.outputQueuedMs = finalOutput.queuedMs
        struct FinalCounters: Encodable { let event = "finalCounters"; let metrics: BenchmarkMetrics }
        try logger.write(FinalCounters(metrics: sample))
        for trace in traces { try logger.write(trace, synchronize: false) }
        struct End: Encodable { let event = "end"; let elapsedSeconds: Double; let stoppedByUser: Bool; let stopReason: String?; let totalSteps: Int; let batteryEnd: Float }
        try logger.write(End(elapsedSeconds: completedSeconds,
                             stoppedByUser: stopFlag.load(ordering: .relaxed), stopReason: stopCause.withLock { $0 }, totalSteps: sample.steps,
                             batteryEnd: await MainActor.run { UIDevice.current.batteryLevel }))
        await finished("Finished — export JSONL below")
    }
}

struct BenchmarkView: View {
    @State private var controller = BenchmarkController()
    @Environment(\.scenePhase) private var scenePhase
    var body: some View {
        Form {
            Section("Configuration") {
                Picker("Mode", selection: $controller.configuration.mode) {
                    Text("Mimi loopback").tag("mimi")
                    Text("Full pipeline throughput (no playback)").tag("throughput")
                    Text("Transformer only").tag("transformer")
                    Text("Mimi + transformer").tag("combined")
                }
                Picker("Transformer", selection: $controller.configuration.modelSelection) {
                    Text("Untrained complete ~1B Q4 + depth").tag("complete-random-q4")
                    Text("1B main Q4 (from Q8)").tag("main-q4")
                    Text("1B main Q6").tag("main-q6")
                    Text("1B main Q8").tag("main-q8")
                    Text("Hibiki-M + depth Q6").tag("speech-q6")
                    Text("Hibiki-M + depth Q4 (from Q6)").tag("speech-q4")
                    Text("Synthetic ~0.5B main Q4").tag("synthetic-7")
                    Text("Synthetic ~0.8B main Q4").tag("synthetic-12")
                    Text("Synthetic ~1.3B main Q4").tag("synthetic-22")
                }.disabled(controller.configuration.mode == "mimi")
                Picker("Duration", selection: $controller.configuration.durationSeconds) {
                    ForEach([60, 300, 600, 1200], id: \.self) { Text("\($0 / 60) min").tag($0) }
                }
                Picker("Context growth limit", selection: $controller.configuration.contextSteps) {
                    Text("Short (16 steps)").tag(16)
                    Text("30 sec (375)").tag(375)
                    Text("2 min (1500)").tag(1500)
                    Text("5 min (3750)").tag(3750)
                }
            }.disabled(controller.running)
            Section("Benchmark") {
                Button(controller.running ? "Stop Benchmark" : "Start Benchmark") {
                    if controller.running { controller.stop() } else { controller.start() }
                }
                Text(controller.status)
                LabeledContent("Steps/sec", value: String(format: "%.2f", controller.latest.stepsPerSecond))
                LabeledContent("Encode / model / decode ms", value: String(format: "%.1f / %.1f / %.1f", controller.latest.encodeMs, controller.latest.transformerMs, controller.latest.decodeMs))
                LabeledContent("Resident MB", value: String(format: "%.0f", Double(controller.latest.residentBytes) / 1e6))
                LabeledContent("Thermal", value: controller.latest.thermalState)
                LabeledContent("Elapsed sec", value: String(format: "%.0f", controller.latest.elapsedSeconds))
                LabeledContent("Dropped / underruns", value: "\(controller.latest.droppedInputFrames) / \(controller.latest.underrunCallbacks)")
                if let url = controller.logURL { ShareLink("Export JSONL", item: url) }
            }
            Section {
                Text("Audio modes are paced at 12.5 frames/sec. Use transformer mode for throughput headroom. Complete/depth models generate eight audio codebooks for playback; main-only models forward microphone tokens. Untrained output is noise, played at reduced gain. Context resets at the chosen limit; this is a growth sweep. Use headphones to avoid acoustic feedback.")
            }
        }
        .navigationTitle("Speech Benchmark")
        .task {
            let args = ProcessInfo.processInfo.arguments
            if let index = args.firstIndex(of: "--benchmark"), args.count > index + 1 {
                controller.configuration.mode = args[index + 1]
                if args.count > index + 2, let seconds = Int(args[index + 2]) { controller.configuration.durationSeconds = seconds }
                if args.count > index + 3 { controller.configuration.modelSelection = args[index + 3] }
                if args.count > index + 4, let steps = Int(args[index + 4]) { controller.configuration.contextSteps = steps }
                if let windowIndex = args.firstIndex(of: "--temporal-window"), args.count > windowIndex + 1 {
                    controller.configuration.temporalWindow = Int(args[windowIndex + 1])
                }
                controller.configuration.optimizedGeneration = !args.contains("--reference-generation")
                controller.configuration.verifyGeneration = args.contains("--verify-generation")
                controller.configuration.halfPrecisionCodec = !args.contains("--fp32-codec")
                controller.configuration.halfPrecisionActivations = !args.contains("--bf16-activations")
                controller.start()
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background && controller.running { controller.stop(reason: "background") }
        }
    }
}
