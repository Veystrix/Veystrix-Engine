import Foundation
import CryptoKit
import Security

/// AES-256-GCM encryption for the local chat history file.
///
/// Private prompts should not be readable as plain text on disk. The key is a
/// 256-bit random value kept in the Keychain; an owner-only file is used only
/// when the Keychain is unavailable (for example an unsigned development
/// build). The file is still 0600 either way.
public enum ChatHistoryCipher {
    public struct Envelope: Codable, Sendable {
        public var version: Int
        public var payload: Data
        public init(version: Int, payload: Data) { self.version = version; self.payload = payload }
    }

    public static let formatVersion = 1

    public static func seal(_ plaintext: Data, key: SymmetricKey) throws -> Data {
        let box = try AES.GCM.seal(plaintext, using: key)
        guard let combined = box.combined else { throw CipherError.sealingFailed }
        return try JSONEncoder().encode(Envelope(version: formatVersion, payload: combined))
    }

    public static func open(_ envelope: Data, key: SymmetricKey) throws -> Data {
        let parsed = try JSONDecoder().decode(Envelope.self, from: envelope)
        guard parsed.version == formatVersion else { throw CipherError.unsupportedVersion }
        let box = try AES.GCM.SealedBox(combined: parsed.payload)
        return try AES.GCM.open(box, using: key)
    }

    public enum CipherError: Error {
        case sealingFailed, unsupportedVersion
    }

    // MARK: key management

    private static let service = "io.github.joyboyreturns.veystrix"
    private static let account = "chat-history-encryption-key"

    /// Existing key or a freshly generated one. Never logs or exports the key.
    public static func loadOrCreateKey(paths: Paths = Paths()) -> SymmetricKey {
        if let data = keychainRead() { return SymmetricKey(data: data) }
        if let data = try? Data(contentsOf: paths.chatKeyFile), data.count == 32 {
            return SymmetricKey(data: data)
        }
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        precondition(status == errSecSuccess, "system random number generator failed")
        let data = Data(bytes)
        if !keychainWrite(data) {
            try? paths.ensureDirectories()
            FileManager.default.createFile(atPath: paths.chatKeyFile.path, contents: data,
                                           attributes: [.posixPermissions: 0o600])
        }
        return SymmetricKey(data: data)
    }

    private static func keychainRead() -> Data? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service,
                                    kSecAttrAccount as String: account,
                                    kSecReturnData as String: true,
                                    kSecMatchLimit as String: kSecMatchLimitOne]
        var out: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &out) == errSecSuccess,
              let data = out as? Data, data.count == 32 else { return nil }
        return data
    }

    private static func keychainWrite(_ data: Data) -> Bool {
        var add: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                  kSecAttrService as String: service,
                                  kSecAttrAccount as String: account,
                                  kSecValueData as String: data,
                                  kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly]
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }
}
