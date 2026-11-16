import Foundation
import Combine

/// View state that survives a relaunch, stored as JSON in UserDefaults.
///
/// Only UI-only preferences belong here (selected tab, composer options).
/// Model settings live in `AppSettings` and conversation text lives in
/// `ChatStore`; neither is duplicated here. A corrupt or missing value falls
/// back to the supplied default instead of breaking the view.
public final class PersistedState<Value: Codable>: ObservableObject {
    @Published public var value: Value {
        didSet { store(value) }
    }

    private let key: String
    private let defaults: UserDefaults
    private let fallback: Value

    public init(_ key: String, default fallback: Value, defaults: UserDefaults = .standard) {
        self.key = key
        self.fallback = fallback
        self.defaults = defaults
        if let data = defaults.data(forKey: key),
           let decoded = try? JSONDecoder().decode(Value.self, from: data) {
            self.value = decoded
        } else {
            self.value = fallback
        }
    }

    private func store(_ value: Value) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        defaults.set(data, forKey: key)
    }
}
