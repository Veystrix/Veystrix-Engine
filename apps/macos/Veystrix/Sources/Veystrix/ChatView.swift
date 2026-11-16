import SwiftUI
import AppKit
import VeystrixKit

private struct ChatMessage: Codable, Identifiable {
    var id = UUID()
    var role: String
    var content: String
    var reasoningContent: String?
    var usage: ChatUsage?
    var finishReason: String?
}

private struct ChatConversation: Codable, Identifiable {
    var id = UUID()
    var model: String
    var messages: [ChatMessage] = []
    var title: String { messages.first(where: { $0.role == "user" }).map { String($0.content.prefix(48)) } ?? "New conversation" }
}

@MainActor private final class ChatStore: ObservableObject {
    @Published var conversations: [ChatConversation] = []
    @Published var selection: UUID?
    @Published private var errors: [UUID: String] = [:]
    @Published private var draftError: String?
    @Published private var tasks: [UUID: Task<Void, Never>] = [:]
    private let file = Paths().applicationSupport.appendingPathComponent("chat-history.json")

    var error: String? {
        get { selection.map { errors[$0] } ?? draftError }
        set {
            if let id = selection { errors[id] = newValue }
            else { draftError = newValue }
        }
    }

    var streaming: Bool { selection.map(isStreaming) ?? false }

    func isStreaming(_ id: UUID) -> Bool { tasks[id] != nil }

    private static let streamSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 7200
        config.timeoutIntervalForResource = 0
        config.waitsForConnectivity = false
        config.connectionProxyDictionary = [:]
        return URLSession(configuration: config)
    }()

    init() {
        if let data = try? Data(contentsOf: file), let saved = try? JSONDecoder().decode([ChatConversation].self, from: data) {
            conversations = saved
            selection = saved.first?.id
        }
    }

    func save(for id: UUID) {
        do {
            try Paths().ensureDirectories()
            let data = try JSONEncoder().encode(conversations)
            try data.write(to: file, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
        } catch { errors[id] = "Could not save chat history: \(error.localizedDescription)" }
    }

    func new(model: String) {
        let c = ChatConversation(model: model)
        conversations.insert(c, at: 0); selection = c.id
        draftError = nil
        save(for: c.id)
    }

    func delete() {
        guard !streaming, let id = selection else { return }
        conversations.removeAll { $0.id == id }; selection = conversations.first?.id
        errors[id] = nil
        save(for: selection ?? id)
        if selection == nil { draftError = errors.removeValue(forKey: id) }
    }

    func stop() {
        if let id = selection { tasks[id]?.cancel() }
    }

    func send(_ text: String,
              model: String,
              port: Int,
              key: String?,
              reasoning: ChatReasoningEffort,
              maxOutputTokens: Int?,
              system: String) {
        guard !streaming, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        if selection == nil { new(model: model) }
        guard let id = selection, let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard conversations[index].model == model else {
            error = "This conversation belongs to \(conversations[index].model). Start a new conversation for \(model)."
            return
        }
        conversations[index].messages.append(ChatMessage(role: "user", content: text))
        let priorMessages = conversations[index].messages.map {
            ChatWireMessage(role: $0.role, content: $0.content, reasoningContent: $0.reasoningContent)
        }
        let reply = ChatMessage(role: "assistant", content: "")
        conversations[index].messages.append(reply)
        errors[id] = nil
        save(for: id)
        tasks[id] = Task {
            defer { tasks[id] = nil; save(for: id) }
            do {
                let wire = ChatRequestBuilder.wireMessages(systemPrompt: system, messages: priorMessages)
                let bodyObj = ChatRequestBuilder.payload(
                    model: model,
                    messages: wire,
                    reasoning: reasoning,
                    maxOutputTokens: maxOutputTokens,
                    stream: true
                )
                let bodyData = try JSONSerialization.data(withJSONObject: bodyObj)
                let request = ChatRequestBuilder.makeURLRequest(port: port, key: key, sessionID: id.uuidString, bodyData: bodyData)
                let (bytes, response) = try await Self.streamSession.bytes(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw NSError(domain: "Veystrix", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid response from engine."])
                }
                guard (200..<300).contains(http.statusCode) else {
                    var errorBytes = Data()
                    for try await byte in bytes {
                        if errorBytes.count < 65536 { errorBytes.append(byte) }
                    }
                    let msg = ChatHTTPErrorParser.message(from: errorBytes, statusCode: http.statusCode)
                    throw NSError(domain: "Veystrix", code: http.statusCode, userInfo: [NSLocalizedDescriptionKey: msg])
                }
                for try await line in bytes.lines {
                    try Task.checkCancellation()
                    let events = ChatSSEParser.parse(line: line)
                    guard !events.isEmpty,
                          let convIdx = conversations.firstIndex(where: { $0.id == id }),
                          let msgIdx = conversations[convIdx].messages.firstIndex(where: { $0.id == reply.id }) else { continue }
                    for event in events {
                        switch event {
                        case .reasoning(let chunk):
                            conversations[convIdx].messages[msgIdx].reasoningContent =
                                (conversations[convIdx].messages[msgIdx].reasoningContent ?? "") + chunk
                        case .content(let chunk):
                            conversations[convIdx].messages[msgIdx].content += chunk
                        case .finishReason(let reason):
                            conversations[convIdx].messages[msgIdx].finishReason = reason
                        case .usage(let usage):
                            conversations[convIdx].messages[msgIdx].usage = usage
                        case .error(let msg):
                            errors[id] = msg
                        case .done:
                            break
                        }
                    }
                }
            } catch {
                errors[id] = Task.isCancelled ? "Generation stopped by user." : error.localizedDescription
            }
        }
    }
}

struct ChatView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var engine: EngineController
    @StateObject private var chat = ChatStore()
    @StateObject private var draft = LocalState("")
    @StateObject private var system = LocalState("")
    @StateObject private var effort = LocalState(ChatReasoningEffort.medium)
    @StateObject private var useExplicitLimit = LocalState(false)
    @StateObject private var explicitLimitText = LocalState("")
    @StateObject private var showOptions = LocalState(false)

    private var activeModelID: String? {
        engine.health?.installedModelID ?? engine.health?.model
    }

    private var currentConversation: ChatConversation? {
        chat.conversations.first { $0.id == chat.selection }
    }

    private var lastUsage: ChatUsage? {
        currentConversation?.messages.last(where: { $0.usage != nil })?.usage
    }

    private var lastFinishReason: String? {
        currentConversation?.messages.last(where: { $0.finishReason != nil })?.finishReason
    }

    private var lastMessageID: UUID? { currentConversation?.messages.last?.id }

    var body: some View {
        HSplitView {
            conversationSidebar
            conversationPane
        }
        .navigationTitle("Chat")
    }

    // MARK: - sidebar

    private var conversationSidebar: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Conversations", systemImage: "bubble.left.and.bubble.right")
                    .font(.headline)
                Spacer()
                Button { if let active = activeModelID { chat.new(model: active) } } label: {
                    Image(systemName: "square.and.pencil")
                }
                .buttonStyle(.borderless)
                .help("New conversation")
                .disabled(activeModelID == nil)
                Button { chat.delete() } label: { Image(systemName: "trash") }
                    .buttonStyle(.borderless)
                    .help("Delete conversation")
                    .disabled(chat.streaming || chat.selection == nil)
            }
            .padding(12)

            Divider()

            if chat.conversations.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "bubble.left.and.bubble.right")
                        .font(.title).foregroundStyle(.secondary)
                    Text("No conversations yet").font(.subheadline)
                    Text("Start one with the pencil above.").font(.caption).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(selection: $chat.selection) {
                    ForEach(chat.conversations) { c in
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "bubble.left.fill")
                                .foregroundStyle(c.model.contains("nemotron") ? Color.purple : Color.blue)
                            VStack(alignment: .leading, spacing: 3) {
                                Text(c.title).font(.subheadline.weight(.medium)).lineLimit(1)
                                HStack(spacing: 4) {
                                    Text(c.model).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                                    if chat.isStreaming(c.id) {
                                        ProgressView().controlSize(.mini)
                                            .accessibilityLabel("Generating")
                                    }
                                }
                            }
                            Spacer(minLength: 0)
                        }
                        .padding(.vertical, 3)
                        .tag(Optional(c.id))
                    }
                }
                .listStyle(.sidebar)
            }
        }
        .frame(minWidth: 190, idealWidth: 230, maxWidth: 280)
    }

    // MARK: - conversation

    private var conversationPane: some View {
        VStack(spacing: 12) {
            chatHeader
            if showOptions.value { optionsCard }
            messagesArea
            if let error = chat.error { NoticeBanner(systemImage: "exclamationmark.triangle.fill", tint: .red, title: "Something went wrong", text: error) }
            composer
        }
        .padding(16)
        .frame(minWidth: 460)
    }

    private var chatHeader: some View {
        HStack(spacing: 12) {
            Image(systemName: activeModelID?.contains("nemotron") == true ? "bolt.horizontal.circle.fill" : "sparkles.rectangle.stack.fill")
                .font(.title2)
                .foregroundStyle(AppTheme.gradient())
            VStack(alignment: .leading, spacing: 2) {
                Text(activeModelID ?? "No model loaded").font(.headline)
                Text(telemetrySummary).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer(minLength: 8)
            if engine.state == .starting {
                ProgressView().controlSize(.small)
                Text("Loading…").font(.caption).foregroundStyle(.secondary)
            }
            Picker("Reasoning", selection: $effort.value) {
                ForEach(ChatReasoningEffort.allCases) { Text($0.label).tag($0) }
            }
            .frame(width: 160)
            Button { showOptions.value.toggle() } label: { Image(systemName: "slider.horizontal.3") }
                .buttonStyle(.borderless)
                .help("Chat options")
        }
        .padding(13)
        .background(AppTheme.cardFill, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(AppTheme.cardStroke, lineWidth: 1))
    }

    private var optionsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Output limit").font(.subheadline.weight(.medium))
                Spacer()
                Picker("", selection: $useExplicitLimit.value) {
                    Text("Unlimited").tag(false)
                    Text("Custom").tag(true)
                }
                .pickerStyle(.segmented).frame(width: 200)
            }
            if useExplicitLimit.value {
                TextField("Max tokens", text: $explicitLimitText.value)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140)
            }
            Text(useExplicitLimit.value
                 ? "A debug cap stops generation early. The engine still clamps it to the room left in the context."
                 : "Unlimited: generation stops on EOS, the context limit, a safety guard, or your Stop button.")
                .font(.caption).foregroundStyle(.secondary)
            Divider()
            TextField("Optional system prompt", text: $system.value, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...3)
        }
        .padding(13)
        .background(AppTheme.cardFill, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(AppTheme.cardStroke, lineWidth: 1))
    }

    private var messagesArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Group {
                    if let c = currentConversation, !c.messages.isEmpty {
                        LazyVStack(spacing: 14) {
                            ForEach(c.messages) { message in
                                messageBubble(message)
                                    .id(message.id)
                            }
                        }
                    } else {
                        EmptyState(systemImage: "message.and.waveform",
                                   title: "A fresh conversation",
                                   text: "Ask the running model anything. Replies stream in as they are generated; reasoning and token usage stay available on each answer.")
                    }
                }
                .padding(.horizontal, 4)
                .padding(.vertical, 8)
            }
            .onChange(of: currentConversation?.messages.count) { _, _ in scrollToLatest(proxy) }
            .onChange(of: lastMessageID) { _, _ in scrollToLatest(proxy) }
            .onAppear { scrollToLatest(proxy) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func scrollToLatest(_ proxy: ScrollViewProxy) {
        guard let id = lastMessageID else { return }
        withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(id, anchor: .bottom) }
    }

    @ViewBuilder
    private func messageBubble(_ message: ChatMessage) -> some View {
        let isUser = message.role == "user"
        HStack(alignment: .top, spacing: 10) {
            if !isUser {
                Image(systemName: "sparkles")
                    .font(.caption)
                    .foregroundStyle(.white)
                    .frame(width: 26, height: 26)
                    .background(AppTheme.gradient(), in: Circle())
            }
            VStack(alignment: isUser ? .trailing : .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(isUser ? "You" : "Assistant").font(.caption.bold()).foregroundStyle(.secondary)
                    if !isUser, let finish = message.finishReason {
                        Text("· \(finish)").font(.caption2).foregroundStyle(.secondary)
                    }
                    if !isUser, let u = message.usage {
                        Text("· \(u.completionTokens) out / \(u.promptTokens) in (\(u.cachedTokens) cached)")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                VStack(alignment: .leading, spacing: 8) {
                    if let reasoning = message.reasoningContent, !reasoning.isEmpty {
                        DisclosureGroup {
                            Text(reasoning)
                                .font(.callout)
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        } label: {
                            Label("Reasoning", systemImage: "brain")
                                .font(.caption.weight(.semibold))
                        }
                    }
                    let placeholder = (message.reasoningContent?.isEmpty == false) ? "Thinking…" : "…"
                    Text(message.content.isEmpty ? placeholder : message.content)
                        .font(.body)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundStyle(message.content.isEmpty ? .secondary : .primary)
                }
                .padding(12)
                .background(
                    (isUser ? AnyShapeStyle(AppTheme.gradient().opacity(0.16)) : AnyShapeStyle(AppTheme.cardFill)),
                    in: RoundedRectangle(cornerRadius: 14, style: .continuous)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .stroke(isUser ? Color.accentColor.opacity(0.22) : AppTheme.cardStroke, lineWidth: 1)
                )
                .frame(maxWidth: 620, alignment: isUser ? .trailing : .leading)

                if !isUser && !message.content.isEmpty {
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(message.content, forType: .string)
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                            .font(.caption2)
                    }
                    .buttonStyle(.borderless)
                }
            }
            .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
            if isUser {
                Image(systemName: "person.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                    .frame(width: 26, height: 26)
            }
        }
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Message the model…", text: $draft.value, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.body)
                .lineLimit(2...6)
                .padding(10)
                .background(AppTheme.cardFill, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(AppTheme.cardStroke, lineWidth: 1))
            if chat.streaming {
                Button(role: .destructive) { chat.stop() } label: {
                    Label("Stop", systemImage: "stop.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.escape, modifiers: [])
            } else {
                Button { submitMessage() } label: {
                    Label("Send", systemImage: "arrow.up.circle.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(engine.state != .running || activeModelID == nil || draft.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
    }

    private func submitMessage() {
        guard let active = activeModelID else { return }
        let parsedLimit: Int?
        do {
            parsedLimit = try ChatRequestBuilder.parseExplicitLimit(
                enabled: useExplicitLimit.value,
                rawText: explicitLimitText.value
            )
        } catch {
            chat.error = "\(error)"
            return
        }
        let text = draft.value
        draft.value = ""
        chat.error = nil
        chat.send(
            text,
            model: active,
            port: model.settings.port,
            key: model.effectiveKey,
            reasoning: effort.value,
            maxOutputTokens: parsedLimit,
            system: system.value
        )
    }

    private var telemetrySummary: String {
        var parts: [String] = []
        let contextLimit = engine.health?.contextLimit
        if let act = engine.health?.activity, act.active == true {
            if act.stage == "prefill", let done = act.prefillDone, let total = act.prefillTotal {
                parts.append("Prefilling \(done)/\(total) tok")
            } else if act.stage == "decode", let dec = act.decodeTokens {
                if let rate = act.decodeTokensPerSecond {
                    parts.append(String(format: "Decoding %d tok (%.1f tok/s)", dec, rate))
                } else {
                    parts.append("Decoding \(dec) tok")
                }
            } else if let stage = act.stage {
                parts.append(stage.capitalized)
            }
        }
        if let u = lastUsage {
            if let limit = contextLimit, limit > 0 {
                let pct = min(100, Int((Double(u.totalTokens) / Double(limit)) * 100.0))
                parts.append("Context \(u.totalTokens)/\(limit) tok (\(pct)%)")
            } else {
                parts.append("Tokens \(u.promptTokens) in + \(u.completionTokens) out = \(u.totalTokens)")
            }
            if u.cachedTokens > 0 { parts.append("Cached \(u.cachedTokens)") }
        } else if let limit = contextLimit, limit > 0 {
            parts.append("Context limit \(limit) tok")
        }
        if let finish = lastFinishReason, !chat.streaming {
            parts.append("Last stop: \(finish)")
        }
        if parts.isEmpty {
            return "Chat uses the running model. Loading another model is controlled in Model."
        }
        return parts.joined(separator: "  ·  ")
    }
}
