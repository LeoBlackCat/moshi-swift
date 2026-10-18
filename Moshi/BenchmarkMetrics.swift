import Foundation
import Darwin
import MLX

struct BenchmarkConfiguration: Encodable, Sendable {
    var mode = "mimi"
    let metalOpsPerBuffer = Int(ProcessInfo.processInfo.environment["MLX_MAX_OPS_PER_BUFFER"] ?? "10") ?? 10
    var optimizedGeneration = true
    var verifyGeneration = false
    var halfPrecisionCodec = true
    var halfPrecisionActivations = true
    var durationSeconds = 60
    var temporalWindow: Int? = nil
    var contextSteps = 375
    var modelSelection = "main-q4"
    let modelRepository = "lmz/moshi-swift"
    let sampleRate = 24000
    let audioFrameRate = 12.5
}

struct BenchmarkMetrics: Codable, Sendable {
    var elapsedSeconds: Double = 0
    var steps = 0
    var stepsPerSecond: Double = 0
    var encodeMs: Double = 0
    var transformerMs: Double = 0
    var decodeMs: Double = 0
    var temporalMs: Double = 0
    var depthMs: Double = 0
    var generatedAudioFrames = 0
    var generatedAudioTokens = 0
    var generatedTokenChecksum: Int64 = 0
    var invalidGeneratedTokens = 0
    var decodedSamples = 0
    var nonfiniteDecodedSamples = 0
    var computeRealtimeFactor: Double = 0
    var frameComputeP50Ms: Double = 0
    var frameComputeP95Ms: Double = 0
    var maxFrameComputeMs: Double = 0
    var computeDeadlineMisses = 0
    var captureToEnqueueMs: Double = 0
    var inputQueuedMs: Double = 0
    var outputQueuedMs: Double = 0
    var firstCaptureToRenderMs: Double?
    var transformerKVBytes = 0
    var mimiKVBytes = 0
    var physicalFootprintBytes: UInt64 = 0
    var residentBytes: UInt64 = 0
    var peakSampledResidentBytes: UInt64 = 0
    var averageSampledResidentBytes: Double = 0
    var mlxActiveBytes = 0
    var mlxCacheBytes = 0
    var mlxPeakBytes = 0
    var cpuPercent: Double = 0
    var thermalState = "unknown"
    var droppedInputFrames = 0
    var rejectedOutputFrames = 0
    var underrunCallbacks = 0
    var missingOutputSamples = 0
    var batteryLevel: Float = -1
    var batteryState = "unknown"
    var contextPosition = 0
    var contextResets = 0
}

struct BenchmarkHeader: Encodable {
    let event = "header"
    let schemaVersion = 5
    let runID: String
    let startedAt: Date
    let device: String
    let ios: String
    let configuration: BenchmarkConfiguration
    let loadSeconds: Double
    let coldWarmupSeconds: Double
    let playbackPrefillSamples: Int
    let modelFileBytes: Int64
    let modelSource: String
    let precision: String
    let logicalParameters: Int
    let includesDepthTransformer: Bool
    let parameterBreakdown: [String: Int]
    let audioTokenSource: String
    let architecture: String
    let inputSource: String
    let generationValidation: String
    let codecValidation: [String: Double]
    let modelWeightTensorBytes: Int
    let mimiWeightTensorBytes: [String: Int]
    let mlxCacheLimitBytes: Int
    let baselineResidentBytes: UInt64
    let afterModelResidentBytes: UInt64
    let afterCodecResidentBytes: UInt64
    let hardwareSampleRate: Double
    let batteryStart: Float
    let startingThermalState: String
    let startingBatteryState: String
    let unavailableMetrics = ["GPU utilization: use Metal System Trace", "power watts: unavailable via public API", "acoustic microphone-to-speaker latency: requires external loopback", "exact disjoint memory attribution", "game memory allowance: requires co-resident game test"]
    let latencyDefinition = "captureToEnqueueMs uses tap arrival minus buffer duration, not acoustic timestamp; firstCaptureToRenderMs ends at source render callback, before hardware output latency"
    let contextPolicy = "temporalWindow if set: bounded sliding frames including current, absolute RoPE; reset only at contextSteps"
}

final class BenchmarkLogger {
    let url: URL
    private let file: FileHandle
    private let encoder = JSONEncoder()
    init() throws {
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Benchmarks")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        url = directory.appendingPathComponent("run-\(UUID().uuidString).jsonl")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        file = try FileHandle(forWritingTo: url)
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
    }
    func write<T: Encodable>(_ value: T) throws {
        var data = try encoder.encode(value); data.append(10)
        try file.write(contentsOf: data)
        try file.synchronize()
    }
    deinit { try? file.close() }
}

func residentMemory() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? UInt64(info.resident_size) : 0
}
func processCPUSeconds() -> Double {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 0 }
    return Double(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) + Double(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6
}
func thermalName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}
func deviceIdentifier() -> String {
    var system = utsname(); uname(&system)
    return withUnsafePointer(to: &system.machine) {
        $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) }
    }
}

func physicalFootprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}
