import SwiftUI
import VeystrixKit

/// Shared visual language for the Veystrix app: soft cards, one accent gradient,
/// quiet labels and clear status. Designed for macOS 14+.
enum AppTheme {
    static let accent = Color.blue
    static let accentAlt = Color.purple

    static let cardFill = Color(nsColor: .controlBackgroundColor)
    static let cardStroke = Color.primary.opacity(0.08)

    static func gradient(for model: ModelFamily? = nil) -> LinearGradient {
        let primary = model == .nemotron35 ? Color.purple : Color.blue
        return LinearGradient(colors: [primary, accentAlt], startPoint: .topLeading, endPoint: .bottomTrailing)
    }
}

/// Consistent page title with an optional icon and subtitle.
struct PageHeader: View {
    let title: String
    let systemImage: String
    var subtitle: String? = nil
    var trailing: AnyView? = nil

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            Image(systemName: systemImage)
                .font(.system(size: 28, weight: .semibold))
                .foregroundStyle(AppTheme.gradient())
                .symbolRenderingMode(.hierarchical)
                .frame(width: 42, height: 42)
                .background(AppTheme.gradient().opacity(0.10), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.largeTitle.bold())
                if let subtitle { Text(subtitle).font(.callout).foregroundStyle(.secondary) }
            }
            Spacer(minLength: 8)
            trailing
        }
    }
}

/// Rounded content card used on every page.
struct Panel<Content: View>: View {
    let title: String
    let systemImage: String
    var subtitle: String? = nil
    var tint: Color = .accentColor
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                    .foregroundStyle(tint.gradient)
                    .font(.subheadline.weight(.semibold))
                Text(title).font(.headline)
            }
            if let subtitle {
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
            content.frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(AppTheme.cardFill, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(AppTheme.cardStroke, lineWidth: 1))
    }
}

/// Compact metric tile.
struct StatCard: View {
    let title: String
    let value: String
    var detail: String? = nil
    var systemImage: String? = nil
    var tint: Color = .accentColor

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                if let systemImage {
                    Image(systemName: systemImage).foregroundStyle(tint.gradient).font(.caption)
                }
                Text(title).font(.caption).foregroundStyle(.secondary)
            }
            Text(value)
                .font(.title3.weight(.semibold))
                .lineLimit(2)
                .minimumScaleFactor(0.8)
                .contentTransition(.numericText())
            if let detail {
                Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(3)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(AppTheme.cardFill, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(AppTheme.cardStroke, lineWidth: 1))
        .accessibilityElement(children: .combine)
    }
}

/// A small colored status dot.
struct StatusDot: View {
    let color: Color
    var body: some View {
        Circle().fill(color.gradient).frame(width: 9, height: 9)
    }
}

/// Friendly empty state for lists and charts.
struct EmptyState: View {
    let systemImage: String
    let title: String
    let text: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 36))
                .foregroundStyle(AppTheme.gradient())
                .symbolRenderingMode(.hierarchical)
            Text(title).font(.headline)
            Text(text).font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
        }
        .frame(maxWidth: .infinity)
        .padding(36)
        .background(AppTheme.cardFill, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(AppTheme.cardStroke, lineWidth: 1))
    }
}

/// Inline notice banner.
struct NoticeBanner: View {
    let systemImage: String
    let tint: Color
    let title: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: systemImage).font(.title3).foregroundStyle(tint.gradient)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(text).font(.callout).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .background(tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(tint.opacity(0.2), lineWidth: 1))
    }
}
