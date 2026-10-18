// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN
import MLXRandom

public struct DepformerConfig {
    var transformer: TransformerConfig
    var numSlices: Int
}

class DepformerSlice: Module {
    @ModuleInfo(key: "emb") var emb: Embedding
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "transformer") var transformer: Transformer

    public init(
        inVocabSize: Int, outVocabSize: Int, mainTransformerDim: Int, cfg: TransformerConfig
    ) {
        self._emb.wrappedValue = Embedding(embeddingCount: inVocabSize, dimensions: cfg.dModel)
        self._linearIn.wrappedValue = Linear(mainTransformerDim, cfg.dModel, bias: false)
        self._linearOut.wrappedValue = Linear(cfg.dModel, outVocabSize, bias: false)
        self._transformer.wrappedValue = Transformer(cfg)
    }
}

class Depformer: Module {
    let cfg: LmConfig
    var transformerCache: [KVCache]
    var compiledExecution = false
    private var compiledSlices: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    private var compiledSampler: Sampler?
    @ModuleInfo(key: "slices") var slices: [DepformerSlice]

    public init(_ cfg: LmConfig, _ cfgDepformer: DepformerConfig, bSize: Int) {
        self.cfg = cfg
        let slices = (0..<cfgDepformer.numSlices).map { idx in
            DepformerSlice(
                inVocabSize: idx == 0 ? cfg.textInVocabSize : cfg.audioVocabSize,
                outVocabSize: cfg.audioVocabSize - 1,
                mainTransformerDim: cfg.transformer.dModel,
                cfg: cfgDepformer.transformer)
        }
        self._slices.wrappedValue = slices
        self.transformerCache = slices[0].transformer.makeCache(bSize: bSize)
    }

    public func sample(
        mainTransformerOut: MLXArray, stepIdx: Int, sampler: Sampler, textToken: MLXArray
    ) -> MLXArray {
        if compiledExecution {
            if compiledSampler !== sampler { compiledSlices.removeAll(); compiledSampler = sampler }
            let bucket = min(stepIdx, cfg.audioDelays.max() ?? 0)
            if compiledSlices[bucket] == nil {
                // Weights are frozen before compilation. Depth KV is local to one audio frame;
                // the only cross-call mutable state is the sampler RNG, explicitly threaded.
                compiledSlices[bucket] = compile(inputs: [MLXRandom.globalState], outputs: [MLXRandom.globalState]) { [unowned self] (inputs: [MLXArray]) -> [MLXArray] in
                    let cache: [KVCache] = self.transformerCache.map { _ in DepthKVCache() }
                    return [self.sampleWithCache(mainTransformerOut: inputs[0], stepIdx: bucket,
                        sampler: sampler, textToken: inputs[1], cache: cache)]
                }
            }
            return compiledSlices[bucket]!([mainTransformerOut, textToken])[0]
        }
        return sampleWithCache(mainTransformerOut: mainTransformerOut, stepIdx: stepIdx,
            sampler: sampler, textToken: textToken, cache: transformerCache)
    }

    private func sampleWithCache(mainTransformerOut: MLXArray, stepIdx: Int, sampler: Sampler,
                                 textToken: MLXArray, cache: [KVCache]) -> MLXArray {
        for c in cache { c.reset() }
        var lastToken = textToken
        var tokens: [MLXArray] = []
        for (sliceIdx, slice) in slices.enumerated() {
            if sliceIdx != 0 && stepIdx < self.cfg.audioDelays[sliceIdx - 1] {
                lastToken = MLXArray([self.cfg.audioPaddingToken()])
            }
            var xs = slice.linearIn(mainTransformerOut) + slice.emb(lastToken)
            xs = slice.transformer(xs, cache: cache)
            let logits = slice.linearOut(xs)
            (lastToken, _) = sampler(logits: logits[0])
            tokens.append(lastToken)
        }
        return concatenated(tokens)
    }
}

public struct LmConfig {
    public var transformer: TransformerConfig
    public var depformer: DepformerConfig?
    public var textInVocabSize: Int
    public var textOutVocabSize: Int
    public var audioVocabSize: Int
    public var audioCodebooks: Int
    public var audioDelays: [Int]

    public func audioEOSToken() -> Int {
        self.audioVocabSize - 2
    }

    public func audioPaddingToken() -> Int {
        self.audioVocabSize - 1
    }

    public func textInitToken() -> Int {
        self.textInVocabSize - 1
    }

    public func depformerSlices() -> Int {
        self.depformer?.numSlices ?? 0
    }

    public static func moshi_2024_07() -> LmConfig {
        let depformer = DepformerConfig(
            transformer:
                TransformerConfig(
                    dModel: 1024,
                    numHeads: 16,
                    numLayers: 6,
                    causal: true,
                    normFirst: true,
                    biasFF: false,
                    biasAttn: false,
                    layerScale: nil,
                    positionalEmbedding: .none,
                    useConvBias: false,
                    gating: true,
                    norm: .rmsNorm,
                    context: 8,
                    maxPeriod: 10000,
                    maxSeqLen: 4096,
                    kvRepeat: 1,
                    dimFeedForward: 1024 * 4,
                    convLayout: false,
                    useRotatingKVCache: false
                ), numSlices: 8)
        return LmConfig(
            transformer: TransformerConfig.v1_7b(),
            depformer: depformer,
            textInVocabSize: 32001,
            textOutVocabSize: 32000,
            audioVocabSize: 2049,
            audioCodebooks: 16,
            audioDelays: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        )
    }

    public static func moshi1b(audioDelay: Int) -> LmConfig {
        let audioDelays = [0] + Array(repeating: audioDelay, count: 7)
        let depformer = DepformerConfig(
            transformer:
                TransformerConfig(
                    dModel: 1024,
                    numHeads: 16,
                    numLayers: 6,
                    causal: true,
                    normFirst: true,
                    biasFF: false,
                    biasAttn: false,
                    layerScale: nil,
                    positionalEmbedding: .none,
                    useConvBias: false,
                    gating: true,
                    norm: .rmsNorm,
                    context: 8,
                    maxPeriod: 10000,
                    maxSeqLen: 4096,
                    kvRepeat: 1,
                    dimFeedForward: 1024 * 4,
                    convLayout: false,
                    useRotatingKVCache: false
                ), numSlices: 8)
        return LmConfig(
            transformer: TransformerConfig.v1_1b(),
            depformer: depformer,
            textInVocabSize: 48001,
            textOutVocabSize: 48000,
            audioVocabSize: 2049,
            audioCodebooks: 16,
            audioDelays: audioDelays + audioDelays
        )
    }

    /// Complete dual-stream architecture sized near one billion total LM parameters.
    /// Random initialization is performed by the benchmark runner, never loaded/trained.
    public static func benchmarkComplete1b() -> LmConfig {
        var cfg = moshi1b(audioDelay: 2)
        cfg.transformer.numLayers = 14
        cfg.textInVocabSize = 8001
        cfg.textOutVocabSize = 8000
        cfg.depformer!.transformer.dModel = 512
        cfg.depformer!.transformer.numHeads = 8
        cfg.depformer!.transformer.dimFeedForward = 2048
        return cfg
    }

    public static func moshi2b(audioDelay: Int) -> LmConfig {
        let audioDelays = [0] + Array(repeating: audioDelay, count: 7)
        let depformer = DepformerConfig(
            transformer:
                TransformerConfig(
                    dModel: 1024,
                    numHeads: 16,
                    numLayers: 6,
                    causal: true,
                    normFirst: true,
                    biasFF: false,
                    biasAttn: false,
                    layerScale: nil,
                    positionalEmbedding: .none,
                    useConvBias: false,
                    gating: true,
                    norm: .rmsNorm,
                    context: 8,
                    maxPeriod: 10000,
                    maxSeqLen: 4096,
                    kvRepeat: 1,
                    dimFeedForward: 1024 * 4,
                    convLayout: false,
                    useRotatingKVCache: false
                ), numSlices: 8)
        return LmConfig(
            transformer: TransformerConfig.v1_2b(),
            depformer: depformer,
            textInVocabSize: 48001,
            textOutVocabSize: 48000,
            audioVocabSize: 2049,
            audioCodebooks: 16,
            audioDelays: audioDelays + audioDelays
        )
    }

    public static func asr300m() -> LmConfig {
        return LmConfig(
            transformer: TransformerConfig.v1_300m(),
            depformer: nil,
            textInVocabSize: 48001,
            textOutVocabSize: 48000,
            audioVocabSize: 2049,
            audioCodebooks: 32,
            audioDelays: Array(repeating: 0, count: 32)
        )
    }

    public static func asr1b() -> LmConfig {
        return LmConfig(
            transformer: TransformerConfig.v1_1b(),
            depformer: nil,
            textInVocabSize: 8001,
            textOutVocabSize: 8000,
            audioVocabSize: 2049,
            audioCodebooks: 32,
            audioDelays: Array(repeating: 0, count: 32)
        )
    }

    public static func asr2b() -> LmConfig {
        return LmConfig(
            transformer: TransformerConfig.v1_2b(),
            depformer: nil,
            textInVocabSize: 4001,
            textOutVocabSize: 4000,
            audioVocabSize: 2049,
            audioCodebooks: 32,
            audioDelays: Array(repeating: 0, count: 32)
        )
    }

    public static func helium2b() -> LmConfig {
        return LmConfig(
            transformer: TransformerConfig.v1_2b(),
            depformer: nil,
            textInVocabSize: 48000,
            textOutVocabSize: 48000,
            audioVocabSize: 2049,
            audioCodebooks: 0,
            audioDelays: Array(repeating: 0, count: 0)
        )
    }
}

public class LM: Module {
    var transformerCache: [KVCache]
    public let cfg: LmConfig
    @ModuleInfo(key: "depformer") var depformer: Depformer?
    @ModuleInfo(key: "transformer") public var transformer: Transformer
    @ModuleInfo(key: "text_emb") var textEmb: Embedding
    @ModuleInfo(key: "out_norm") var outNorm: UnaryLayer
    @ModuleInfo(key: "text_linear") var textLinear: Linear
    @ModuleInfo(key: "audio_embs") var audioEmbs: [Embedding]

    public init(_ cfg: LmConfig, bSize: Int) {
        self.cfg = cfg
        self._transformer.wrappedValue = Transformer(cfg.transformer)
        self._depformer.wrappedValue = cfg.depformer.map { Depformer(cfg, $0, bSize: bSize) }
        self._textEmb.wrappedValue = Embedding(
            embeddingCount: cfg.textInVocabSize, dimensions: cfg.transformer.dModel)
        self._outNorm.wrappedValue =
            switch cfg.transformer.norm {
            case .layerNorm:
                LayerNorm(dimensions: cfg.transformer.dModel, eps: 1e-5)
            case .rmsNorm: RMSNorm(dimensions: cfg.transformer.dModel, eps: 1e-8)
            }
        self._textLinear.wrappedValue = Linear(
            cfg.transformer.dModel, cfg.textOutVocabSize, bias: false)
        self._audioEmbs.wrappedValue = (0..<cfg.audioCodebooks).map { _ in
            Embedding(embeddingCount: cfg.audioVocabSize, dimensions: cfg.transformer.dModel)
        }
        self.transformerCache = self._transformer.wrappedValue.makeCache(bSize: bSize)
    }

    /// Rebuild empty temporal caches after loading/casting weights, before any generation.
    public func rebuildTemporalCache(dtype: DType) {
        precondition(transformerCache.allSatisfy { $0.offset == 0 })
        transformerCache = transformer.makeCache(bSize: 1, dtype: dtype)
    }

    /// Storage/scheduling optimization only; all depth layers and attention positions remain.
    public func useCompactDepthCache(_ enabled: Bool, compiled: Bool = false) {
        guard let depformer else { return }
        transformer.useCompiledFeedForward(compiled)
        depformer.compiledExecution = compiled
        depformer.transformerCache = enabled
            ? depformer.transformerCache.map { _ in DepthKVCache() }
            : depformer.slices[0].transformer.makeCache(bSize: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = textEmb(x)
        x = transformer(x, cache: self.transformerCache)
        return textLinear(outNorm(x))
    }

    public func resetCache() {
        for cache in self.transformerCache {
            cache.reset()
        }
    }

    public func stepMain(textIds: MLXArray?, audioIds: [MLXArray]) -> (MLXArray, MLXArray) {
        var x = textIds.flatMap { textEmb($0) }
        for (a, emb) in zip(audioIds, self.audioEmbs) {
            let e = emb(a)
            x = x.map { $0 + e } ?? e
        }
        let out = outNorm(transformer(x!, cache: self.transformerCache))
        let logits = textLinear(out[0..., -1, 0...])
        return (out, logits)
    }

    public func sample(
        textIds: MLXArray?,
        audioIds: [MLXArray],
        stepIdx: Int,
        textSampler: Sampler,
        audioSampler: Sampler,
        cb: Callbacks
    ) -> (MLXArray, MLXArray) {
        cb.onEvent(.beginStep)
        var x = textIds.flatMap { textEmb($0) }
        for (a, emb) in zip(audioIds, self.audioEmbs) {
            let e = emb(a)
            x = x.map { $0 + e } ?? e
        }
        let mainTransformerOut = outNorm(transformer(x!, cache: self.transformerCache))
        let textLogits = textLinear(mainTransformerOut[0..., -1, 0...])
        let (textToken, _) = textSampler(logits: textLogits)
        textToken.eval()
        cb.onEvent(.endStep)
        if let depformer = self.depformer {
            cb.onEvent(.beginDepformer)
            let audioTokens = depformer.sample(
                mainTransformerOut: mainTransformerOut,
                stepIdx: stepIdx,
                sampler: audioSampler,
                textToken: textToken)
            audioTokens.eval()
            cb.onEvent(.endDepformer)
            return (textToken, audioTokens)
        } else {
            return (textToken, MLXArray())
        }
    }

    public func warmup() {
        let sampler = Sampler()
        let textIds = MLXArray.zeros([1, 1], dtype: .int32)
        let audioIds = (0..<self.cfg.depformerSlices()).map { _ in
            MLXArray.zeros([1, 1], dtype: .int32)
        }
        let (textToken, audioTokens) = sample(
            textIds: textIds, audioIds: audioIds, stepIdx: 0, textSampler: sampler,
            audioSampler: sampler, cb: EmptyCallbacks())
        eval(textToken)
        eval(audioTokens)
        for c in self.transformerCache {
            c.reset()
        }
    }
}

let zeroToken: Int = -1
let ungeneratedToken: Int = -2

public class LMGen {
    let model: LM
    let maxSteps: Int
    let audioSampler: Sampler
    let textSampler: Sampler
    let numCodebooks: Int
    let genSequence: MLXArray
    let mainCodebooks: Int
    let cb: Callbacks
    var stepIdx: Int
    private let hostHistory: Bool
    private var history: [Int32]

    public init(
        _ model: LM, maxSteps: Int, audioSampler: Sampler, textSampler: Sampler,
        cb: Callbacks = EmptyCallbacks(), hostHistory: Bool = false
    ) {
        self.hostHistory = hostHistory
        self.history = hostHistory ? Array(repeating: Int32(ungeneratedToken), count: (1 + model.cfg.audioCodebooks) * maxSteps) : []
        self.model = model
        self.maxSteps = maxSteps
        self.audioSampler = audioSampler
        self.textSampler = textSampler
        self.numCodebooks = 1 + model.cfg.audioCodebooks
        self.genSequence = MLXArray.full(
            [1, self.numCodebooks, maxSteps], values: MLXArray(ungeneratedToken, dtype: .int32))
        self.stepIdx = 0
        self.mainCodebooks = self.model.cfg.depformerSlices()
        self.cb = cb
    }

    public func step(otherAudioTokens: MLXArray) -> MLXArray? {
        if hostHistory { return stepHost(otherAudioTokens: otherAudioTokens) }
        if self.stepIdx >= self.maxSteps {
            return nil
        }
        let textIds: MLXArray
        if self.stepIdx == 0 {
            textIds = MLXArray([self.model.cfg.textOutVocabSize]).reshaped([1, 1])
        } else {
            textIds = self.genSequence[.newAxis, 0..., 0, self.stepIdx - 1]
        }
        self.genSequence[0..., (1 + self.mainCodebooks)..., self.stepIdx] = otherAudioTokens
        var audioIds: [MLXArray] = []
        for (cbIdx, delay) in self.model.cfg.audioDelays.enumerated() {
            let genIdx = self.stepIdx - 1 - delay
            let audioToken: MLXArray
            if genIdx >= 0 {
                audioToken = self.genSequence[.newAxis, 0..., 1 + cbIdx, genIdx]
            } else {
                audioToken = MLXArray([self.model.cfg.audioPaddingToken()]).reshaped([1, 1])
            }
            if (audioToken .== MLXArray(ungeneratedToken)).any().item<Bool>() {
                fatalError("ungenerated value in audio tokens, cb \(cbIdx), step \(stepIdx)")
            }
            assert(audioToken.shape == [1, 1])
            audioIds.append(audioToken)
        }
        if (textIds .== MLXArray(ungeneratedToken)).any().item<Bool>() {
            fatalError("ungenerated value in text tokens, step \(stepIdx)")
        }
        assert(textIds.shape == [1, 1])
        let (tt, at) = model.sample(
            textIds: textIds,
            audioIds: audioIds,
            stepIdx: self.stepIdx,
            textSampler: self.textSampler,
            audioSampler: self.audioSampler,
            cb: self.cb
        )
        assert(tt.shape == [1])
        assert(at.shape == [self.model.cfg.depformerSlices()])

        self.genSequence[0..., 0, self.stepIdx] = tt
        for cbIdx in 0..<self.model.cfg.depformerSlices() {
            let delay = self.model.cfg.audioDelays[cbIdx]
            let genIdx = self.stepIdx - delay
            if genIdx >= 0 {
                self.genSequence[0..., cbIdx + 1, genIdx] = at[cbIdx]
            }
        }

        self.stepIdx += 1
        return textIds
    }

    private func stepHost(otherAudioTokens: MLXArray) -> MLXArray? {
        guard stepIdx < maxSteps else { return nil }
        let other = otherAudioTokens.asArray(Int32.self)
        precondition(other.count == model.cfg.audioCodebooks - mainCodebooks)
        for (index, token) in other.enumerated() {
            history[(1 + mainCodebooks + index) * maxSteps + stepIdx] = token
        }
        let text = stepIdx == 0 ? Int32(model.cfg.textOutVocabSize) : history[stepIdx - 1]
        precondition(text != Int32(ungeneratedToken))
        let textIds = MLXArray([text]).reshaped([1, 1])
        let audioIds = model.cfg.audioDelays.enumerated().map { index, delay in
            let source = stepIdx - 1 - delay
            let token = source < 0 ? Int32(model.cfg.audioPaddingToken()) : history[(1 + index) * maxSteps + source]
            precondition(token != Int32(ungeneratedToken))
            return MLXArray([token]).reshaped([1, 1])
        }
        let (tt, at) = model.sample(textIds: textIds, audioIds: audioIds, stepIdx: stepIdx,
            textSampler: textSampler, audioSampler: audioSampler, cb: cb)
        history[stepIdx] = tt.item(Int32.self)
        let audio = at.asArray(Int32.self)
        precondition(audio.count == mainCodebooks)
        for (index, token) in audio.enumerated() {
            let target = stepIdx - model.cfg.audioDelays[index]
            if target >= 0 { history[(1 + index) * maxSteps + target] = token }
        }
        stepIdx += 1
        return textIds
    }

    public func lastAudioTokens() -> MLXArray? {
        if hostHistory {
            let index = stepIdx - 1 - (model.cfg.audioDelays.max() ?? 0)
            guard index >= 0 else { return nil }
            let tokens = (0..<mainCodebooks).map { history[(1 + $0) * maxSteps + index] }
            precondition(!tokens.contains(Int32(ungeneratedToken)))
            guard !tokens.contains(Int32(model.cfg.audioPaddingToken())) else { return nil }
            return MLXArray(tokens).reshaped([1, mainCodebooks])
        }
        let genIdx = self.stepIdx - 1 - (self.model.cfg.audioDelays.max() ?? 0)
        if genIdx < 0 {
            return nil
        }
        let tokens = self.genSequence[0..., 1...self.mainCodebooks, genIdx]
        if (tokens .== MLXArray(ungeneratedToken)).any().item<Bool>() {
            fatalError("ungenerated value in text tokens, step \(stepIdx)")
        }
        if (tokens .== MLXArray(self.model.cfg.audioPaddingToken())).any().item<Bool>() {
            return nil
        }
        return tokens
    }

    public func reset() {
        self.stepIdx = 0
        self.model.resetCache()
        if hostHistory { history = Array(repeating: Int32(ungeneratedToken), count: history.count) }
        else { self.genSequence[0...] = MLXArray(ungeneratedToken) }
        cb.onReset()
    }
}
