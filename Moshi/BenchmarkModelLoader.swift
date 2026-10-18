// Model loading adapted from Kyutai's MIT-licensed reference Evaluator.
// Kept independent of SwiftUI and MainActor for benchmark worker execution.
import Foundation
import Hub
import MLX
import MLXNN
import MoshiLib

final class BenchmarkModelLoader {
    func downloadFromHub(id: String, filename: String) async throws -> URL {
        let directory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let api = HubApi(downloadBase: directory)
        let repo = Hub.Repo(id: id)
        let cached = api.localRepoLocation(repo).appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: cached.path) { return cached }
        let base = try await api.snapshot(from: repo, matching: filename)
        return base.appendingPathComponent(filename)
    }
    func makeMoshi(_ url: URL, _ cfg: LmConfig) throws -> LM {
        let weights = try loadArrays(url: url)
        let parameters = ModuleParameters.unflattened(weights)
        let model = LM(cfg, bSize: 1)
        if url.lastPathComponent.hasSuffix(".q4.safetensors") {
            quantize(model: model, groupSize: 32, bits: 4)
        } else if url.lastPathComponent.hasSuffix(".q6.safetensors") {
            quantize(model: model, groupSize: 64, bits: 6)
        } else if url.lastPathComponent.hasSuffix(".q8.safetensors") {
            quantize(model: model, groupSize: 64, bits: 8)
        }
        try model.update(parameters: parameters, verify: [.all])
        eval(model)
        return model
    }
    func makeMimi(numCodebooks: Int, halfPrecision: Bool = false, contiguousWeights: Bool = !ProcessInfo.processInfo.arguments.contains("--reference-codec-layout")) async throws -> Mimi {
        let cfg = MimiConfig.mimi_2024_07(numCodebooks: numCodebooks)
        let model = Mimi(cfg, bSize: 1)

        let url = try await downloadFromHub(
            id: "lmz/moshi-swift",
            filename: "tokenizer-dbaa9758-checkpoint125.safetensors")
        let origWeights = try loadArrays(url: url)
        var weights: [String: MLXArray] = [:]
        for (var key, var weight) in origWeights {
            // Mutating the keys while iterating over the map seems pretty dodgy, not sure what the idiomatic
            // way to do this is in swift. Hopefully this is copy on write and it's all good :)
            if key.hasPrefix("encoder.model") {
                key.replace("encoder.model.", with: "encoder.")
            }
            if key.hasPrefix("decoder.model") {
                key.replace("decoder.model.", with: "decoder.")
            }
            if key.hasSuffix(".in_proj_weight") {
                key.replace(".in_proj_weight", with: ".in_proj.weight")
            }
            if key.hasSuffix(".linear1.weight") {
                key.replace(".linear1.weight", with: ".gating.linear1.weight")
            }
            if key.hasSuffix(".linear2.weight") {
                key.replace(".linear2.weight", with: ".gating.linear2.weight")
            }
            // Awfully hardcoded matching between the pytorch layers and their mlx equivalent :(
            for (layerIdx, decoderIdx) in [2, 5, 8, 11].enumerated() {
                key.replace("decoder.\(decoderIdx).", with: "decoder.layers.\(layerIdx).upsample.")
                key.replace(
                    "decoder.\(decoderIdx + 1).", with: "decoder.layers.\(layerIdx).residuals.0.")
            }
            for (layerIdx, encoderIdx) in [1, 4, 7, 10].enumerated() {
                key.replace(
                    "encoder.\(encoderIdx).", with: "encoder.layers.\(layerIdx).residuals.0.")
                key.replace(
                    "encoder.\(encoderIdx + 2).", with: "encoder.layers.\(layerIdx).downsample.")
            }
            key.replace("decoder.0.", with: "decoder.init_conv1d.")
            key.replace("decoder.14.", with: "decoder.final_conv1d.")
            key.replace("encoder.0.", with: "encoder.init_conv1d.")
            key.replace("encoder.14.", with: "encoder.final_conv1d.")
            key.replace(".block.1.", with: ".block.0.")
            key.replace(".block.3.", with: ".block.1.")
            // PyTorch layout for conv weights is outC, inC, kSize, for MLX it's outC, kSize, inC
            if key.hasSuffix(".conv.weight") || key.hasSuffix(".output_proj.weight")
                || key.hasSuffix(".input_proj.weight")
            {
                weight = weight.swappedAxes(-1, -2)
            }
            // PyTorch layout for conv-transposed weights is inC, outC, kSize, for MLX it's outC, kSize, inC
            if key.hasSuffix(".convtr.weight") {
                weight = weight.transposed(axes: [1, 2, 0])
            }

            // print(key, weight.shape)
            weights[key] = weight
        }
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])
        if halfPrecision {
            // Keep EMA counts, codebook sums, nearest-neighbor lookup and projections in FP32.
            // These accumulators can exceed FP16's finite range.
            model.update(parameters: ModuleParameters.unflattened(model.parameters().flattened().map { name, value in
                (name, name.hasPrefix("quantizer.") ? value : value.asType(.float16))
            }))
            model.inferenceDType = .float16
            model.useCompiledFeedForward(true)
        }
        if contiguousWeights {
            // Materialize the PyTorch-to-MLX convolution layout once. The pinned
            // Metal convolution backend otherwise copies noncontiguous filters per call.
            let filters = model.parameters().flattened().filter { name, _ in
                name.hasSuffix(".conv.weight") || name.hasSuffix(".convtr.weight")
                    || name.hasSuffix(".input_proj.weight") || name.hasSuffix(".output_proj.weight")
            }
            eval(filters.map { $0.1 })
            var noncontiguousBytes = 0
            for (_, value) in filters {
                var expected = 1
                for axis in value.shape.indices.reversed() {
                    if value.dim(axis) > 1 && value.strides[axis] != expected { noncontiguousBytes += value.nbytes; break }
                    expected *= value.dim(axis)
                }
            }
            model.update(parameters: ModuleParameters.unflattened(filters.map { ($0.0, asStrided($0.1)) }))
            eval(model)
            print("BENCHMARK contiguous codec: materialized \(noncontiguousBytes) noncontiguous filter bytes")
        }
        return model
    }

    func verifyCodecExecution(_ candidate: Mimi) async throws -> [String: Double] {
        let reference = try await makeMimi(numCodebooks: 8, halfPrecision: true, contiguousWeights: false)
        eval(reference); candidate.resetState()
        var tokenMismatches = 0, maxPCMError: Float = 0
        for frame in 0..<64 {
            if frame == 32 { reference.resetState(); candidate.resetState() }
            let values: [Float] = (0..<1920).map { index in
                let t = Double(frame * 1920 + index) / 24000
                return Float(0.08 * sin(2 * .pi * 173 * t) + 0.03 * sin(2 * .pi * 431 * t))
            }
            let pcm = StreamArray(MLXArray(values).reshaped([1, 1, 1920]))
            let refCodes = reference.encodeStep(pcm).asArray()!.asArray(Int32.self)
            let codes = candidate.encodeStep(pcm).asArray()!.asArray(Int32.self)
            tokenMismatches += zip(refCodes, codes).filter { $0 != $1 }.count
            let input = StreamArray(MLXArray(refCodes).reshaped([1, 8, 1]))
            let expected = reference.decodeStep(input).asArray()!.asArray(Float.self)
            let actual = candidate.decodeStep(input).asArray()!.asArray(Float.self)
            guard actual.count == 1920 && actual.allSatisfy({ $0.isFinite }) else { throw CustomError("Invalid optimized codec PCM") }
            for (a, b) in zip(expected, actual) { maxPCMError = max(maxPCMError, abs(a - b)) }
        }
        candidate.resetState()
        guard tokenMismatches == 0 && maxPCMError == 0 else { throw CustomError("Codec execution parity failed: tokens=\(tokenMismatches), PCM max=\(maxPCMError)") }
        let result = ["executionParityFrames": 64.0, "executionTokenMismatches": Double(tokenMismatches), "executionMaxPCMError": Double(maxPCMError)]
        print("BENCHMARK codec execution parity: \(result)")
        return result
    }

    func verifyCodecPrecision(_ candidate: Mimi) async throws -> [String: Double] {
        let reference = try await makeMimi(numCodebooks: 8, contiguousWeights: false)
        eval(reference); candidate.resetState()
        var squaredError = 0.0, squaredReference = 0.0
        var matches = 0, tokenCount = 0, sampleCount = 0
        for frame in 0..<32 {
            let values: [Float] = (0..<1920).map { index in
                let t = Double(frame * 1920 + index) / 24000
                return Float(0.08 * sin(2 * .pi * 173 * t) + 0.03 * sin(2 * .pi * 431 * t))
            }
            let pcm = StreamArray(MLXArray(values).reshaped([1, 1, 1920]))
            let refCodes = reference.encodeStep(pcm).asArray()!.asArray(Int32.self)
            let newCodes = candidate.encodeStep(pcm).asArray()!.asArray(Int32.self)
            guard refCodes.count == 8 && newCodes.count == 8 && newCodes.allSatisfy({ $0 >= 0 && $0 < 2048 }) else {
                throw CustomError("FP16 codec produced invalid encoded tokens")
            }
            matches += zip(refCodes, newCodes).filter { $0 == $1 }.count; tokenCount += 8
            // Feed identical codes to isolate decoder precision from discrete encoder decisions.
            let codes = StreamArray(MLXArray(refCodes).reshaped([1, 8, 1]))
            let ref = reference.decodeStep(codes).asArray()!.asArray(Float.self)
            let actual = candidate.decodeStep(codes).asArray()!.asArray(Float.self)
            guard ref.count == 1920 && actual.count == ref.count && actual.allSatisfy({ $0.isFinite }) else {
                throw CustomError("FP16 codec produced invalid decoded PCM")
            }
            for (a, b) in zip(ref, actual) {
                squaredError += pow(Double(a) - Double(b), 2); squaredReference += pow(Double(a), 2)
            }
            sampleCount += ref.count
        }
        let relativeRMS = sqrt(squaredError / max(squaredReference, 1e-20))
        candidate.resetState()
        guard relativeRMS < 0.02 else { throw CustomError("FP16 decoder relative RMS error too high: \(relativeRMS)") }
        let result = ["frames": 32.0, "decodedSamples": Double(sampleCount),
                      "encoderTokenAgreement": Double(matches) / Double(tokenCount), "decoderRelativeRMSError": relativeRMS]
        print("BENCHMARK codec parity: \(result)")
        return result
    }
}
