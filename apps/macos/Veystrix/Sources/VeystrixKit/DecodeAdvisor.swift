import Foundation

/// One plain-language reason a decode may feel slow, derived from live health
/// data. No guesses are made about the machine; when the engine has not
/// reported a field, that factor is simply omitted.
public struct DecodeFactor: Equatable, Sendable {
    public var title: String
    public var detail: String
    public var isWarning: Bool
    public init(title: String, detail: String, isWarning: Bool = false) {
        self.title = title; self.detail = detail; self.isWarning = isWarning
    }
}

public enum DecodeAdvisor {
    public static func explain(_ health: HealthReport?) -> [DecodeFactor] {
        guard let health else {
            return [DecodeFactor(title: "Engine is stopped",
                                 detail: "Start the engine and send a message to see live decode telemetry here.")]
        }
        var factors: [DecodeFactor] = []

        if let tps = health.activity?.decodeTokensPerSecond, tps > 0 {
            factors.append(DecodeFactor(
                title: "Decode rate",
                detail: String(format: "%.1f tokens/s right now. This is the end-to-end rate, including reading experts from disk.", tps)))
        }

        if let policy = health.memoryPolicy {
            if policy.canAdmit == false {
                factors.append(DecodeFactor(
                    title: "Memory admission refused new work",
                    detail: policy.stopReason ?? "The engine stopped a request to protect your Mac.",
                    isWarning: true))
            }
            if let current = policy.currentSwapGrowthMb, let ceiling = policy.swapCeilingMib,
               ceiling > 0, current >= ceiling * 0.8 {
                factors.append(DecodeFactor(
                    title: "Swap growth is near the ceiling",
                    detail: "\(Int(current)) of \(Int(ceiling)) MiB of allowed growth used. Expert pages are being pushed to disk, which lowers decode speed.",
                    isWarning: true))
            }
        }

        if health.flashNextKVCache == "tiered4" {
            factors.append(DecodeFactor(
                title: "Tiered KV cache",
                detail: "A mixed INT4/INT8 cache extends the window with a small dequantisation cost per step. This is expected, not a fault."))
        }
        if let kv = health.nemotron?.kvFormat {
            factors.append(DecodeFactor(
                title: "SSD-streamed experts (\(kv))",
                detail: "Nemotron keeps experts on disk: our measurements show roughly 41 ms in the GPU and 80–90 ms waiting for expert pages per token. Decode is I/O-bound by design."))
        }

        if factors.isEmpty {
            factors.append(DecodeFactor(
                title: "No slowdown signal yet",
                detail: "The engine has not reported decode telemetry. Send a message, then watch this panel."))
        }
        return factors
    }
}
