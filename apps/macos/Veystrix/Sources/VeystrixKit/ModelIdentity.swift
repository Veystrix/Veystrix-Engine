import Foundation

/// App-side mirror of the engine's model-identity descriptor
/// (`veystrix/model_identity.py`). Facts the app and engine must agree on live
/// here, so swapping the default model touches one file instead of a dozen
/// scattered strings. Values are identical to the current baseline.
public enum ModelIdentity {
    public struct Family: Sendable, Equatable {
        public let title: String
        public let modelTypes: Set<String>
        public let servedID: String
        public let downloadRepo: String
        public let contextLimit: Int
    }

    public static let flashNext = Family(
        title: "Qwen 3.8 Flash-Next",
        modelTypes: ["qwen4_exp", "qwen4_exp_text"],
        servedID: "veystrix-flashnext",
        downloadRepo: "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP",
        contextLimit: 262_144
    )

    public static let nemotron = Family(
        title: "NVIDIA Nemotron 3.5 Lightning (experimental)",
        modelTypes: ["nemotron_h"],
        servedID: "veystrix-nemotron35-lightning",
        downloadRepo: "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit",
        contextLimit: 1_048_576
    )
}
