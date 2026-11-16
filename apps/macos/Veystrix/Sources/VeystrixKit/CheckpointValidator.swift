import Foundation

/// Result of checking a model folder. Validation reads only small JSON metadata files and `stat`s the weight
/// shards. It never opens or reads a weight file, so it cannot load a model or use significant memory.
public struct CheckpointReport: Equatable, Sendable {
    public enum Verdict: Equatable, Sendable { case compatible, incomplete, incompatible, notFound }
    public var verdict: Verdict
    public var path: String
    public var modelType: String?
    public var shardCount: Int = 0
    public var totalBytes: Int64 = 0
    public var missingFiles: [String] = []
    public var notes: [String] = []
    /// Where the MTP draft tensors live: "none", "sidecar" or "indexed".
    public var mtpSource: String = "none"
    public var mtpIssues: [String] = []

    public var isUsable: Bool { verdict == .compatible }
    /// Approximate disk size, for display.
    public var humanSize: String { ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file) }
}

/// Mirrors the engine's own rules (`veystrix/checkpoints.py: flashnext_compatible`) so the app and the engine agree.
public enum CheckpointValidator {
    static let tokenizerFiles = ["tokenizer.json", "tokenizer_config.json"]
    static let supportedModelTypes: Set<String> = ModelFamily.flashnext.modelTypes

    /// Mirrors `models/flashnext/checkpoint_compat.py::REQUIRED_MTP_SUBSTRINGS`.
    static let requiredMTPSubstrings = [
        "language_model.mtp.pre_fc_norm_embedding.weight",
        "language_model.mtp.pre_fc_norm_hidden.weight",
        "language_model.mtp.fc_embedding.weight",
        "language_model.mtp.fc_hidden.weight",
        "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
        "language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight",
        "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight",
    ]
    private static let mtpPrefix = "language_model.mtp."
    private static let mtpSidecar = "model-mtp.safetensors"

    public static func validate(directory: URL, family: ModelFamily = .flashnext,
                                fileManager: FileManager = .default) -> CheckpointReport {
        var report = CheckpointReport(verdict: .notFound, path: directory.path)
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDir), isDir.boolValue else {
            report.notes.append("The folder does not exist.")
            return report
        }
        report.verdict = .incompatible

        let config = readJSON(directory.appendingPathComponent("config.json"))
        if config == nil { report.missingFiles.append("config.json") }
        for name in tokenizerFiles where !fileManager.fileExists(atPath: directory.appendingPathComponent(name).path) {
            report.missingFiles.append(name)
        }

        var types = Set<String>()
        if let config {
            if let t = config["model_type"] as? String { types.insert(t) }
            for key in ["text_config", "llm_config"] {
                if let nested = config[key] as? [String: Any], let t = nested["model_type"] as? String { types.insert(t) }
            }
            report.modelType = types.sorted().joined(separator: ", ")
            if types.isDisjoint(with: family.modelTypes) {
                report.notes.append("Unsupported model type (\(report.modelType ?? "unknown")) for \(family.title). Choose the matching model above, or a folder with a supported checkpoint.")
                return report
            }
            if family == .nemotron35 {
                // Only the MLX affine 4-bit conversion is supported (NVIDIA's NVFP4 is a different format).
                let q = config["quantization"] as? [String: Any]
                guard q?["mode"] as? String == "affine", q?["bits"] as? Int == 4 else {
                    report.notes.append("Nemotron needs the MLX affine 4-bit conversion (mlx-community). NVFP4 and other formats are not supported.")
                    return report
                }
            }
        }

        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard let index = readJSON(indexURL), let weightMap = index["weight_map"] as? [String: Any], !weightMap.isEmpty else {
            report.missingFiles.append("model.safetensors.index.json")
            report.verdict = .incomplete
            report.notes.append("The weight index is missing or empty.")
            return report
        }
        let shardNames = Set(weightMap.values.compactMap { $0 as? String })
        guard shardNames.count == weightMap.count || !shardNames.isEmpty, shardNames.allSatisfy(isSaneShardName) else {
            report.verdict = .incomplete
            report.missingFiles.append("model.safetensors.index.json (invalid shard names)")
            report.notes.append("The weight index names a file that is not a plain .safetensors file name. It is refused for safety.")
            return report
        }
        report.shardCount = shardNames.count
        let mtpIsFatal = CheckpointValidator.inspectMTP(directory: directory, weightMap: weightMap,
                                                        fileManager: fileManager, report: &report)
        if mtpIsFatal { return report }
        if let leftovers = try? fileManager.contentsOfDirectory(atPath: directory.path),
           leftovers.contains(where: { $0.hasSuffix(".incomplete") || $0.hasSuffix(".part") }) {
            report.verdict = .incomplete
            report.missingFiles.append("partial download files present")
            report.notes.append("A download is still in progress or was interrupted. Finish it, then validate again.")
            return report
        }
        for shard in shardNames.sorted() {
            let url = directory.appendingPathComponent(shard)
            if let attrs = try? fileManager.attributesOfItem(atPath: url.path), let size = attrs[.size] as? NSNumber {
                report.totalBytes += size.int64Value
            } else {
                report.missingFiles.append(shard)
            }
        }
        if !report.missingFiles.isEmpty {
            report.verdict = .incomplete
            report.notes.append("\(report.missingFiles.count) required file(s) are missing. Finish or repair the download, then validate again.")
        } else {
            report.verdict = .compatible
            report.notes.append("All \(report.shardCount) weight files are present. Their contents were not read.")
        }
        return report
    }

    /// Detect the MTP draft source without reading any weights. A checkpoint
    /// with both sidecar and indexed MTP is refused, as in the engine.
    /// Returns true when the verdict is fatal and validation should stop.
    static func inspectMTP(directory: URL, weightMap: [String: Any],
                           fileManager: FileManager, report: inout CheckpointReport) -> Bool {
        let sidecar = fileManager.fileExists(atPath: directory.appendingPathComponent(mtpSidecar).path)
        let indexed = Set(weightMap.keys.filter { $0.hasPrefix(mtpPrefix) })
        if sidecar && !indexed.isEmpty {
            report.verdict = .incompatible
            report.notes.append("Both \(mtpSidecar) and indexed MTP tensors are present; refusing to pick one. Remove one copy, then validate again.")
            return true
        }
        if sidecar {
            report.mtpSource = "sidecar"
            return false
        }
        guard !indexed.isEmpty else { return false }
        report.mtpSource = "indexed"
        let missing = requiredMTPSubstrings.filter { !indexed.contains($0) }
        report.mtpIssues = missing
        if !missing.isEmpty {
            report.notes.append("MTP draft block is incomplete (\(missing.count) of \(requiredMTPSubstrings.count) tensors missing). Speculative decoding needs a complete block; the engine may disable it.")
        }
        return false
    }

    /// The model family a folder belongs to, read from its `config.json` only. Nil when it matches none.
    public static func detectFamily(directory: URL) -> ModelFamily? {
        guard let config = readJSON(directory.appendingPathComponent("config.json")) else { return nil }
        var types = Set<String>()
        if let t = config["model_type"] as? String { types.insert(t) }
        for key in ["text_config", "llm_config"] {
            if let nested = config[key] as? [String: Any], let t = nested["model_type"] as? String { types.insert(t) }
        }
        return ModelFamily.allCases.first { !$0.modelTypes.isDisjoint(with: types) }
    }

    /// Same constraints as the engine: a plain `*.safetensors` file name, no separators, not hidden, not absolute.
    static func isSaneShardName(_ name: String) -> Bool {
        name.hasSuffix(".safetensors") && !name.contains("/") && !name.contains("\\")
            && name != "." && name != ".." && !name.hasPrefix(".") && !(name as NSString).isAbsolutePath
    }

    private static func readJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe), data.count < 64 * 1024 * 1024 else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    /// Plain-language licensing and compatibility reminder shown next to a checkpoint.
    public static let licenseNotice = """
    Veystrix does not include a model and does not download one. A checkpoint is licensed by its publisher, \
    not by Veystrix. Read that license before you download, use, modify or share the checkpoint. \
    Supported by this build: the Qwen3.8 Flash-Next family in MLX 4-bit format \
    (architecture \(ModelIdentity.flashNext.modelTypes.sorted().joined(separator: " / "))), and, \
    experimentally, NVIDIA Nemotron 3.5 Lightning in the mlx-community affine 4-bit conversion \
    (architecture \(ModelIdentity.nemotron.modelTypes.sorted().joined(separator: " / ")); \
    NVIDIA Open Model / OpenMDW 1.1 license). The engine verifies the Nemotron layout before it loads.
    """
}
