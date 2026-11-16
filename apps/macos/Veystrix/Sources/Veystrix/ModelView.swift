import SwiftUI
import AppKit
import VeystrixKit

struct ModelView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var engine: EngineController

    private let qwenModelURL = URL(string: "https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")!
    private let nemotronModelURL = URL(string: "https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit")!

    private var isNemotron: Bool { model.settings.selectedModel == .nemotron35 }

    /// The two controls below are computed per model, so they need explicit bindings.
    private var contextBinding: Binding<Int> {
        Binding(
            get: { model.settings.activeContextTokens },
            set: { model.settings.activeContextTokens = min(max(1, $0), model.settings.activeContextLimit) }
        )
    }

    private var maxTokensBinding: Binding<Int> {
        Binding(
            get: { max(0, model.settings.activePerformance.defaultOutputTokens) },
            set: { model.settings.activePerformance.defaultOutputTokens = max(0, $0) }
        )
    }

    private var contextPresets: [(String, Int)] {
        isNemotron
            ? [("64K", 65_536), ("256K", 262_144), ("1M", 1_048_576)]
            : [("32K", 32_768), ("128K", 131_072), ("256K", 262_144)]
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                hero

                section("Model", systemImage: "cpu", subtitle: "Pick which local model the engine runs.") {
                    Picker("Model", selection: Binding(
                        get: { model.settings.selectedModel },
                        set: { model.selectModel($0) }
                    )) {
                        ForEach(ModelFamily.allCases) { Text($0.title).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(engine.state == .starting || engine.state == .stopping || engine.isExternallyManaged)

                    HStack(spacing: 8) {
                        StatusDot(color: isNemotron ? .purple : .blue)
                        Text(isNemotron
                             ? "Nemotron 3.5 Lightning — experimental, needs the mlx-community affine 4-bit checkpoint."
                             : "Qwen 3.8 Flash-Next — the default model, SSD-streamed experts in MLX 4-bit.")
                            .font(.callout).foregroundStyle(.secondary)
                    }

                    if engine.state == .running {
                        Divider()
                        LabeledContent("Running", value: engine.health?.model ?? "unknown")
                        Button("Switch to Selected Model…") { model.requestRestart() }
                            .buttonStyle(.borderedProminent)
                            .disabled(engine.isExternallyManaged || engine.health?.busy != false || model.checkpoint?.isUsable != true)
                        Text("Switching stops the app-managed engine and discards its live model cache. Finish active harness work first.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                section("KV cache & context", systemImage: "memorychip",
                        subtitle: "Applies to \(model.settings.selectedModel.title). Restart to use new values.") {
                    if isNemotron {
                        Picker("KV cache precision", selection: $model.settings.nemotronKVCache) {
                            ForEach(NemotronKVCache.allCases) { Text($0.title).tag($0) }
                        }
                        .pickerStyle(.segmented)
                        .onChange(of: model.settings.nemotronKVCache) { _, precision in
                            model.settings.nemotronContextTokens = min(model.settings.nemotronContextTokens, precision.contextLimit)
                        }
                    } else {
                        Picker("KV cache precision", selection: $model.settings.flashNextKVCache) {
                            ForEach(FlashNextKVCache.allCases) { Text($0.title).tag($0) }
                        }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Context window").font(.subheadline.weight(.medium))
                            Spacer()
                            TextField("tokens", value: contextBinding, format: .number)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 120)
                                .accessibilityLabel("Context window in tokens")
                        }
                        HStack(spacing: 8) {
                            ForEach(contextPresets, id: \.0) { preset in
                                Button(preset.0) { model.settings.activeContextTokens = preset.1 }
                                    .buttonStyle(.bordered)
                                    .controlSize(.small)
                            }
                            Spacer()
                            Text("max \(model.settings.activeContextLimit.formatted())")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Max output tokens").font(.subheadline.weight(.medium))
                            Spacer()
                            TextField("0 = uncapped", value: maxTokensBinding, format: .number)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 120)
                                .accessibilityLabel("Maximum output tokens")
                        }
                        Text("0 keeps generation uncapped; the engine clamps a request to the room left in the window.")
                            .font(.caption).foregroundStyle(.secondary)
                    }

                    if isNemotron, let status = engine.health?.nemotron {
                        Divider()
                        LabeledContent("Admitted context", value: status.admittedContext.formatted() + " tokens")
                        LabeledContent("Attention KV format", value: status.kvFormat)
                        LabeledContent("1M context validation", value: status.millionTokenValidation ? "Passed" : "Pending")
                    }
                }

                section("Model folder", systemImage: "folder", subtitle: "Each model keeps its own folder.") {
                    LabeledContent("Folder") {
                        Text(model.settings.activeCheckpointPath.isEmpty
                             ? "None chosen"
                             : model.paths.abbreviate(model.settings.activeCheckpointPath))
                            .textSelection(.enabled)
                            .foregroundStyle(model.settings.activeCheckpointPath.isEmpty ? .secondary : .primary)
                    }
                    HStack {
                        Button("Choose Folder…") { choose() }.disabled(engine.state.isActive)
                        Button("Validate Again") { model.revalidateCheckpoint() }
                            .disabled(model.settings.activeCheckpointPath.isEmpty)
                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([
                                URL(fileURLWithPath: model.settings.activeCheckpointPath)
                            ])
                        }.disabled(model.settings.activeCheckpointPath.isEmpty)
                    }
                }

                if let c = model.checkpoint {
                    section("Validation", systemImage: c.isUsable ? "checkmark.seal.fill" : "xmark.octagon.fill",
                            subtitle: "Weights are never read.") {
                        HStack(spacing: 8) {
                            StatusDot(color: c.isUsable ? .green : .red)
                            Text(c.verdictText.capitalized).font(.headline)
                        }
                        LabeledContent("Architecture", value: c.modelType ?? "unknown")
                        LabeledContent("Weight files", value: "\(c.shardCount)")
                        LabeledContent("Disk space used", value: c.humanSize)
                        if !c.missingFiles.isEmpty {
                            LabeledContent("Missing") {
                                Text(c.missingFiles.prefix(8).joined(separator: ", ")).foregroundStyle(.red)
                            }
                        }
                        ForEach(c.notes, id: \.self) {
                            Text($0).font(.callout).foregroundStyle(.secondary)
                        }
                    }
                }

                section("Disk", systemImage: "internaldrive", subtitle: "Keep room for the KV cache and snapshots.") {
                    let snap = MemoryProbe.snapshot()
                    LabeledContent("Free disk space",
                                   value: snap.cacheVolumeFreeBytes.map {
                                       ByteCountFormatter.string(fromByteCount: $0, countStyle: .file)
                                   } ?? "unknown")
                    Text("Keep at least 20 GB free beyond the model and the persistent cache quota (\(model.settings.cacheQuotaGB) GB by default).")
                        .font(.callout).foregroundStyle(.secondary)
                }

                section("Supported models", systemImage: "sparkles",
                        subtitle: "Only these two models are supported in this release.") {
                    Link("Qwen 3.8 Flash-Next — MLX 4-bit MTP", destination: qwenModelURL)
                    Link("NVIDIA Nemotron 3.5 Lightning — MLX 4-bit", destination: nemotronModelURL)
                    Text("Veystrix does not download models. Use hf download with the repository name from the link, then choose the downloaded folder above.")
                        .font(.callout).foregroundStyle(.secondary)
                    Divider()
                    Text(CheckpointValidator.licenseNotice).font(.callout).foregroundStyle(.secondary)
                }
            }
            .padding(20)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .navigationTitle("Model")
    }

    // MARK: - building blocks

    private func section<Content: View>(_ title: String, systemImage: String, subtitle: String? = nil,
                                        @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            if let subtitle {
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 10) { content() }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
                .background(.background, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(.quaternary, lineWidth: 1)
                )
        }
    }

    private var hero: some View {
        let accent = isNemotron ? Color.purple : Color.blue
        return HStack(spacing: 14) {
            Image(systemName: isNemotron ? "bolt.horizontal.circle.fill" : "sparkles.rectangle.stack.fill")
                .font(.system(size: 34))
                .foregroundStyle(accent.gradient)
                .symbolRenderingMode(.hierarchical)
            VStack(alignment: .leading, spacing: 3) {
                Text(model.settings.selectedModel.title)
                    .font(.title2.bold())
                Text(engine.state == .running ? "Engine running" : engine.state.label)
                    .font(.subheadline).foregroundStyle(.secondary)
            }
            Spacer()
            if model.checkpoint?.isUsable == true {
                Label("Ready", systemImage: "checkmark.circle.fill")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.green)
            }
        }
        .padding(16)
        .background(
            LinearGradient(colors: [accent.opacity(0.16), accent.opacity(0.04)],
                           startPoint: .topLeading, endPoint: .bottomTrailing),
            in: RoundedRectangle(cornerRadius: 16, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(accent.opacity(0.25), lineWidth: 1)
        )
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Choose the folder that contains the model (config.json, tokenizer files and *.safetensors)."
        if panel.runModal() == .OK, let url = panel.url { model.chooseCheckpoint(url) }
    }
}

private struct StatusDot: View {
    let color: Color
    var body: some View {
        Circle().fill(color.gradient).frame(width: 9, height: 9)
    }
}
