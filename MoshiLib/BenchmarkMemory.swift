import MLX
import MLXNN

// Logical tensor storage, not disjoint physical resident pages or allocator overhead.
public extension LM {
    var benchmarkKVCacheBytes: Int {
        let caches = transformerCache + (depformer?.transformerCache ?? [])
        return caches.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
    }
}
public extension Mimi {
    var benchmarkKVCacheBytes: Int {
        (encoderCache + decoderCache).flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
    }
    var benchmarkWeightBytes: [String: Int] {
        func bytes(_ modules: [Module]) -> Int {
            modules.flatMap { $0.parameters().flattened() }.reduce(0) { $0 + $1.1.nbytes }
        }
        return ["encoder": bytes([encoder, encoderTransformer, downsample]),
                "decoder": bytes([decoder, decoderTransformer, upsample]),
                "quantizer": bytes([quantizer])]
    }
}
