import Foundation
import CryptoKit

/// Detects accidental or deliberate changes to the bundled engine and app
/// resources by comparing SHA-256 digests against a signed-manifest file.
///
/// Limits (important): this is a tamper *detector*, not copy protection. A
/// attacker who can write the app bundle can also remove the check or forge the
/// manifest. It exists to catch casual modification and corrupted downloads.
/// A Developer ID signature + notarization is what makes Gatekeeper trust it.
public enum BundleIntegrity {
    public static let manifestName = "BundleIntegrity.json"
    public static let version = 1

    public struct Manifest: Codable, Equatable, Sendable {
        public var version: Int
        public var files: [String: String]   // relative path -> hex SHA-256

        public init(version: Int = BundleIntegrity.version, files: [String: String] = [:]) {
            self.version = version
            self.files = files
        }
    }

    public enum Result: Equatable, Sendable {
        case notBundled          // no manifest: development / source-checkout build
        case intact
        case tampered([String])  // changed, missing or unexpected files
    }

    /// Files whose digests are pinned. Bundled app code and small text resources
    /// only: user settings, logs, signed binaries and model folders are excluded.
    /// Only files whose bytes are stable across code signing. Mach-O binaries
    /// (Sparkle, the embedded Python runtime) are signed after this manifest
    /// is generated, so hashing them here would create false tamper reports.
    public static let protectedPrefixes = ["engine/lib/", "Info.plist", "VERSION", "LICENSE.txt",
                                           "THIRD_PARTY_NOTICES.md"]
    // Signing writes BUILD-CHANNEL and re-signs Mach-O files after the manifest
    // is generated. Pinning them would make every signed app look tampered.
    private static let excludedNames: Set<String> = ["BUILD-CHANNEL"]
    private static let excludedSuffixes = [".dylib", ".so"]

    public static func sha256(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Walk a resources directory and pin every file under the protected set.
    public static func createManifest(resources: URL, fileManager: FileManager = .default) throws -> Manifest {
        var files: [String: String] = [:]
        let enumerator = fileManager.enumerator(at: resources, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            guard let relative = relativePath(url, under: resources), isProtected(relative) else { continue }
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue else { continue }
            guard let data = fileManager.contents(atPath: url.path) else { continue }
            files[relative] = sha256(data)
        }
        return Manifest(files: files)
    }

    /// Compare the on-disk resources with the manifest.
    public static func verify(resources: URL, fileManager: FileManager = .default) -> Result {
        guard let data = fileManager.contents(atPath: resources.appendingPathComponent(manifestName).path),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data),
              manifest.version == version else { return .notBundled }
        let current = (try? createManifest(resources: resources, fileManager: fileManager))?.files ?? [:]
        var issues = current.filter { manifest.files[$0.key] != $0.value }.map(\.key)
        issues += manifest.files.keys.filter { current[$0] == nil }
        return issues.isEmpty ? .intact : .tampered(issues.sorted())
    }

    /// Write a manifest into a resources directory (used by packaging, not by app start).
    public static func writeManifest(resources: URL, fileManager: FileManager = .default) throws {
        let manifest = try createManifest(resources: resources, fileManager: fileManager)
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: resources.appendingPathComponent(manifestName), options: .atomic)
    }

    private static func isProtected(_ relative: String) -> Bool {
        relative != manifestName && !excludedNames.contains(relative)
            && !excludedSuffixes.contains { relative.hasSuffix($0) }
            && protectedPrefixes.contains { relative == $0 || relative.hasPrefix($0) }
    }

    private static func relativePath(_ url: URL, under root: URL) -> String? {
        let base = root.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(base + "/") else { return nil }
        return String(path.dropFirst(base.count + 1))
    }
}
