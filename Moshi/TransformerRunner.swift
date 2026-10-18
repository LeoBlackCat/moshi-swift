import Foundation
import MLX
import MLXNN
import MLXRandom
import MoshiLib

/// Inference-only workload. No tokenizer, ASR, text rendering, or training.
final class TransformerRunner {
    let model: LM
    let generator: LMGen?
    let optimizedGeneration: Bool
    var generationValidation = "not requested"
    let parameterCount: Int
    let sourceFileBytes: Int64
    let sourceName: String
    let precision: String
    let includesDepthTransformer: Bool
    var parameterBreakdown: [String: Int] = [:]
    private let timing = GenerationTiming()
    var temporalMs: Double { timing.temporalMs }
    var depthMs: Double { timing.depthMs }
    private var lastText = MLXArray.zeros([1, 1], dtype: .int32)
    private let sampler = Sampler()
    private var position = 0
    private let contextLimit: Int

    init(selection: String, contextLimit: Int, loader: BenchmarkModelLoader, optimizedGeneration: Bool = true, halfPrecisionActivations: Bool = true, temporalWindow: Int? = nil) async throws {
        self.optimizedGeneration = optimizedGeneration
        self.contextLimit = contextLimit
        var cfg = selection == "complete-random-q4" ? LmConfig.benchmarkComplete1b() : (selection == "speech-q6" || selection == "speech-q4") ? LmConfig.moshi1b(audioDelay: 2) : LmConfig.asr1b()
        includesDepthTransformer = (selection == "speech-q6" || selection == "speech-q4") || selection == "complete-random-q4"
        if let temporalWindow {
            guard includesDepthTransformer, temporalWindow == 1500 else { throw CustomError("Unsupported bounded temporal window") }
            cfg.transformer.context = temporalWindow
            cfg.transformer.useRotatingKVCache = true
        }
        // Explicitly synthetic main-transformer scaling; not trained conversational models.
        if selection.hasPrefix("synthetic-") || selection == "complete-random-q4" {
            let layers = Int(selection.split(separator: "-").last!) ?? 8
            if !includesDepthTransformer { cfg.transformer.numLayers = layers }
            MLXRandom.seed(42)
            model = LM(cfg, bSize: 1)
            parameterCount = model.parameters().flattened().reduce(0) { $0 + $1.1.size }
            for (name, tensor) in model.parameters().flattened() {
                let group = name.hasPrefix("depformer.") ? "depth" : name.hasPrefix("transformer.") ? "temporal" : "embeddings_and_heads"
                parameterBreakdown[group, default: 0] += tensor.size
            }
            model.update(parameters: ModuleParameters.unflattened(model.parameters().flattened().map {
                ($0.0, $0.1.asType(.bfloat16))
            }))
            sourceFileBytes = 0
            sourceName = includesDepthTransformer ? "random-complete-moshi-1b-seed42" : "random-initialized-main-\(layers)-layers"
            precision = "Q4 group64 (synthetic)"
            Self.quantizeIncrementally(model, bits: 4)
        } else {
            let filename = (selection == "speech-q6" || selection == "speech-q4") ? "moshi-37c6cfd6@200.q6.safetensors" :
                selection == "main-q6" ? "moshi-70f8f0ea@500.q6.safetensors" : "moshi-70f8f0ea@500.q8.safetensors"
            let url = try await loader.downloadFromHub(id: "lmz/moshi-swift", filename: filename)
            sourceFileBytes = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? 0
            sourceName = filename
            model = try loader.makeMoshi(url, cfg)
            // Count logical weights, not packed uint32 storage or quantization metadata.
            parameterCount = Self.logicalParameterCount(model)
            if selection == "main-q4" || selection == "speech-q4" {
                precision = selection == "speech-q4" ? "Q4 group64 (requantized from Q6)" : "Q4 group64 (requantized from Q8)"
                let originalMapping = Self.logicalParameterMapping(model)
                Self.quantizeIncrementally(model, bits: 4)
                guard Self.logicalParameterMapping(model) == originalMapping,
                      Self.logicalParameterCount(model) == parameterCount else {
                    throw CustomError("Requantization changed parameter mapping")
                }
                print("BENCHMARK mapping: PASS: all leaf paths and logical shapes preserved, \(parameterCount) parameters")
            } else { precision = selection.hasSuffix("q6") ? "Q6 group64" : "Q8 group64" }
        }
        if halfPrecisionActivations {
            // Keep packed Q4/Q6/Q8 integers unchanged; use native FP16 for affine scales,
            // normalization and activations instead of this MLX build's emulated BF16.
            model.update(parameters: ModuleParameters.unflattened(model.parameters().flattened().map { name, value in
                (name, value.dtype.isFloatingPoint ? value.asType(.float16) : value)
            }))
        }
        eval(model)
        if temporalWindow != nil {
            model.rebuildTemporalCache(dtype: halfPrecisionActivations ? .float16 : .bfloat16)
        }
        model.useCompactDepthCache(optimizedGeneration, compiled: optimizedGeneration)
        generator = includesDepthTransformer ? LMGen(model, maxSteps: contextLimit + 1, audioSampler: sampler, textSampler: sampler, cb: timing, hostHistory: optimizedGeneration) : nil
    }

    /// Exercise the real quantized model, identical input and RNG seeds, through delay/reset boundaries.
    /// Fail before benchmarking if optimized token routing or cache changes alter sampled tokens.
    func verifyGenerationParity() throws {
        if model.cfg.transformer.useRotatingKVCache {
            guard verifyRotatingCache(window: model.cfg.transformer.context) else { throw CustomError("Sliding cache boundary verification failed") }
            print("BENCHMARK sliding cache: PASS: 1500 slots, two wraps, absolute offsets, causal visibility and reset reuse")
        }
        guard includesDepthTransformer else { throw CustomError("Parity check requires a complete model") }
        func trace(optimized: Bool) -> [[Int32]] {
            model.useCompactDepthCache(optimized, compiled: optimized)
            model.resetCache()
            MLXRandom.seed(1234)
            let gen = LMGen(model, maxSteps: 33, audioSampler: sampler, textSampler: sampler, hostHistory: optimized)
            var result: [[Int32]] = []
            for step in 0..<64 {
                if step == 32 { gen.reset() }
                let input = MLXArray((0..<8).map { Int32((step * 71 + $0 * 19) % 2048) }).reshaped([1, 8])
                let text = gen.step(otherAudioTokens: input)!.asArray(Int32.self)
                let audio = gen.lastAudioTokens()?.asArray(Int32.self) ?? []
                result.append(text + audio)
            }
            return result
        }
        let reference = trace(optimized: false)
        let optimized = trace(optimized: true)
        model.useCompactDepthCache(optimizedGeneration, compiled: optimizedGeneration)
        reset()
        MLXRandom.seed(42)
        guard reference == optimized else {
            let mismatch = zip(reference, optimized).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? -1
            throw CustomError("Generation parity failed at step \(mismatch)")
        }
        generationValidation = "PASS: 64 seeded stochastic steps, text + all 8 audio codebooks, including reset and delay boundaries"
        print("BENCHMARK parity: \(generationValidation)")
    }

    private static func logicalParameterMapping(_ model: Module) -> [String: [Int]] {
        var result: [String: [Int]] = [:]
        for (path, layer) in model.leafModules().flattened() {
            if let q = layer as? QuantizedLinear {
                result[path + ".weight"] = [q.weight.dim(0), q.weight.dim(1) * 32 / q.bits]
                if let bias = q.bias { result[path + ".bias"] = bias.shape }
            } else if let q = layer as? QuantizedEmbedding {
                result[path + ".weight"] = [q.weight.dim(0), q.weight.dim(1) * 32 / q.bits]
            } else {
                for (name, tensor) in layer.parameters().flattened() { result[path + "." + name] = tensor.shape }
            }
        }
        return result
    }

    private static func logicalParameterCount(_ model: Module) -> Int {
        model.leafModules().flattened().reduce(0) { total, pair in
            if let layer = pair.1 as? QuantizedLinear {
                return total + layer.weight.size * 32 / layer.bits + (layer.bias?.size ?? 0)
            }
            if let layer = pair.1 as? QuantizedEmbedding { return total + layer.weight.size * 32 / layer.bits }
            return total + pair.1.parameters().flattened().reduce(0) { $0 + $1.1.size }
        }
    }
    private static func quantizeIncrementally(_ model: Module, bits: Int) {
        var updates: [(String, Module)] = []
        for (path, layer) in model.leafModules().flattened() {
            let replacement: Module?
            if let q = layer as? QuantizedLinear {
                let weight = dequantized(q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize, bits: q.bits)
                replacement = QuantizedLinear(weight: weight, bias: q.bias, groupSize: 64, bits: bits)
            } else if let q = layer as? QuantizedEmbedding {
                let weight = dequantized(q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize, bits: q.bits)
                replacement = QuantizedEmbedding(weight: weight, groupSize: 64, bits: bits)
            } else { replacement = quantizeSingle(layer: layer, groupSize: 64, bits: bits) }
            if let replacement {
                eval(replacement)
                updates.append((path, replacement))
            }
        }
        // Apply the complete tree once: partial array updates replace/truncate arrays in MLX 0.21.
        model.update(modules: ModuleChildren.unflattened(updates))
        GPU.clearCache()
    }
    func reset() {
        generator?.reset(); model.resetCache(); position = 0
        lastText = MLXArray.zeros([1, 1], dtype: .int32)
    }
    @discardableResult
    func step(codes: MLXArray?) -> MLXArray? {
        var generated: MLXArray?
        if let generator {
            let tokens = codes.map { $0[0..., 0..<8, 0] } ?? MLXArray.zeros([1, 8], dtype: .int32)
            generator.step(otherAudioTokens: tokens)?.eval()
            generated = generator.lastAudioTokens()?.reshaped([1, 8, 1])
            generated?.eval()
        } else {
            let audio = (0..<model.cfg.audioCodebooks).map { index in
                if let codes, index < codes.dim(1) { return codes[0..., index, 0].reshaped([1, 1]) }
                return MLXArray.zeros([1, 1], dtype: .int32)
            }
            let (text, _) = model.sample(textIds: lastText, audioIds: audio, stepIdx: position,
                                         textSampler: sampler, audioSampler: sampler, cb: EmptyCallbacks())
            text.eval(); lastText = text.reshaped([1, 1])
        }
        position += 1
        return generated
    }
}

/// Upstream callbacks bracket evaluated temporal and all eight sequential depth slices.
private final class GenerationTiming: Callbacks {
    var temporalMs = 0.0
    var depthMs = 0.0
    private var started = 0.0
    func onReset() { temporalMs = 0; depthMs = 0 }
    func onInputAudioTokens(_ codes: MLXArray) {}
    func onOutputTextToken(_ token: Int) {}
    func onOutputAudioTokens(_ codes: MLXArray) {}
    func onEvent(_ eventKind: EventKind) {
        let now = ProcessInfo.processInfo.systemUptime
        switch eventKind {
        case .beginStep, .beginDepformer: started = now
        case .endStep: temporalMs = (now - started) * 1000
        case .endDepformer: depthMs = (now - started) * 1000
        default: break
        }
    }
}
