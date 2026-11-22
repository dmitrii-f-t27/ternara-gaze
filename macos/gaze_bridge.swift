import AppKit
import ApplicationServices
import AVFoundation
import CoreGraphics
import Foundation
import Vision

let EYE_W = 32
let EYE_H = 16

func screenBounds() -> CGRect {
    return CGDisplayBounds(CGMainDisplayID())
}

func currentMouseU16() -> (Int, Int) {
    let b = screenBounds()
    guard let ev = CGEvent(source: nil) else { return (32768, 32768) }
    let pt = ev.location
    let nx = max(0.0, min(1.0, Double((pt.x - b.origin.x) / max(1.0, b.size.width))))
    let ny = max(0.0, min(1.0, Double((pt.y - b.origin.y) / max(1.0, b.size.height))))
    return (Int(nx * 65535.0), Int(ny * 65535.0))
}

func queryMagneticTargetRect(screenPt: CGPoint) -> CGRect? {
    guard AXIsProcessTrusted() else { return nil }
    let sysWide = AXUIElementCreateSystemWide()
    var elem: AXUIElement?
    let err = AXUIElementCopyElementAtPosition(sysWide, Float(screenPt.x), Float(screenPt.y), &elem)
    guard err == .success, let el = elem else { return nil }

    var posVal: CFTypeRef?
    var sizeVal: CFTypeRef?
    guard AXUIElementCopyAttributeValue(el, kAXPositionAttribute as CFString, &posVal) == .success,
          AXUIElementCopyAttributeValue(el, kAXSizeAttribute as CFString, &sizeVal) == .success,
          let pv = posVal, let sv = sizeVal,
          CFGetTypeID(pv) == AXValueGetTypeID(),
          CFGetTypeID(sv) == AXValueGetTypeID() else {
        return nil
    }
    var pt = CGPoint.zero
    var sz = CGSize.zero
    AXValueGetValue(pv as! AXValue, .cgPoint, &pt)
    AXValueGetValue(sv as! AXValue, .cgSize, &sz)

    // Only magnetically snap to reasonably sized UI controls (buttons, Dock icons, tabs, links, fields)
    if sz.width >= 16 && sz.width <= 340 && sz.height >= 16 && sz.height <= 125 {
        return CGRect(origin: pt, size: sz)
    }
    return nil
}

func atomicClickAtPoint(target: CGPoint, right: Bool) {
    let b = screenBounds()
    let prevPt = CGEvent(source: nil)?.location ?? CGPoint(x: b.midX, y: b.midY)

    let downType: CGEventType = right ? .rightMouseDown : .leftMouseDown
    let upType: CGEventType = right ? .rightMouseUp : .leftMouseUp
    let btn: CGMouseButton = right ? .right : .left

    CGWarpMouseCursorPosition(target)
    usleep(8_000)
    if let down = CGEvent(mouseEventSource: nil, mouseType: downType, mouseCursorPosition: target, mouseButton: btn) {
        down.post(tap: .cghidEventTap)
    }
    usleep(18_000)
    if let up = CGEvent(mouseEventSource: nil, mouseType: upType, mouseCursorPosition: target, mouseButton: btn) {
        up.post(tap: .cghidEventTap)
    }
    usleep(8_000)
    CGWarpMouseCursorPosition(prevPt)
}

func u16ToScreenPoint(xU16: Int, yU16: Int) -> CGPoint {
    let b = screenBounds()
    let nx = Double(max(0, min(65535, xU16))) / 65535.0
    let ny = Double(max(0, min(65535, yU16))) / 65535.0
    return CGPoint(x: b.origin.x + nx * b.size.width, y: b.origin.y + ny * b.size.height)
}

// MARK: - Smooth 60 FPS Floating Second Cursor + Magnetic Region Lock + Nose/Eye HD HUD

final class GazeOverlayView: NSView {
    var mode: String = "intro"   // "intro" | "calibrate" | "gaze"
    var targetX: CGFloat = 0.5
    var targetY: CGFloat = 0.5
    var renderX: CGFloat = 0.5
    var renderY: CGFloat = 0.5
    var holdPct: CGFloat = 0.0
    var regionLocked: Bool = false
    var snappedRect: CGRect? = nil  // in screen top-left coordinates

    var calPct: CGFloat = 0.0
    var calState: String = "AIM" // "AIM" | "REC" | "BLINK"
    var calLabel: String = ""

    var introStep: String = "3"
    var introSubtitle: String = "Камера HD 720p фиксирует нос и оба глаза"

    var clickFlash: CGFloat = 0.0
    var clickWasRight: Bool = false
    var pulsePhase: CGFloat = 0.0

    // Live Nose + Eye Lock telemetry (purely in RAM, zero video saved)
    var eyeLocked: Bool = false
    var leftEyeU4: [UInt8] = [UInt8](repeating: 0, count: EYE_W * EYE_H)
    var rightEyeU4: [UInt8] = [UInt8](repeating: 0, count: EYE_W * EYE_H)
    var pupLX: Int = 0
    var pupLY: Int = 0
    var pupRX: Int = 0
    var pupRY: Int = 0
    var noseX: Int = 0
    var noseY: Int = 0
    var earL: Int = 0
    var earR: Int = 0

    override var isFlipped: Bool { return true }

    private func drawEyeCrop(
        rect: NSRect,
        crop: [UInt8],
        pupX: Int,
        pupY: Int,
        ear: Int,
        title: String
    ) {
        var pixels = [UInt8](repeating: 0, count: EYE_W * EYE_H)
        if crop.count == EYE_W * EYE_H {
            for i in 0..<(EYE_W * EYE_H) {
                pixels[i] = crop[i] << 4
            }
        }
        let colorSpace = CGColorSpaceCreateDeviceGray()
        if let provider = CGDataProvider(data: Data(pixels) as CFData),
           let cgImg = CGImage(
               width: EYE_W,
               height: EYE_H,
               bitsPerComponent: 8,
               bitsPerPixel: 8,
               bytesPerRow: EYE_W,
               space: colorSpace,
               bitmapInfo: CGBitmapInfo(rawValue: 0),
               provider: provider,
               decode: nil,
               shouldInterpolate: true,
               intent: .defaultIntent
           ) {
            if let ctx = NSGraphicsContext.current?.cgContext {
                ctx.saveGState()
                ctx.translateBy(x: rect.minX, y: rect.maxY)
                ctx.scaleBy(x: 1.0, y: -1.0)
                ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: rect.width, height: rect.height))
                ctx.restoreGState()
            }
        }

        let lockColor = eyeLocked
            ? NSColor(calibratedRed: 0.15, green: 0.96, blue: 0.55, alpha: 0.95)
            : NSColor(calibratedRed: 1.0, green: 0.35, blue: 0.25, alpha: 0.95)
        let border = NSBezierPath(roundedRect: rect, xRadius: 5, yRadius: 5)
        lockColor.setStroke()
        border.lineWidth = 2.0
        border.stroke()

        let normX = 0.5 + CGFloat(pupX) / 256.0
        let normY = 0.5 + CGFloat(pupY) / 256.0
        let cx = rect.minX + max(0.1, min(0.9, normX)) * rect.width
        let cy = rect.minY + max(0.1, min(0.9, normY)) * rect.height

        let cross = NSBezierPath()
        cross.move(to: NSPoint(x: cx - 7, y: cy))
        cross.line(to: NSPoint(x: cx + 7, y: cy))
        cross.move(to: NSPoint(x: cx, y: cy - 7))
        cross.line(to: NSPoint(x: cx, y: cy + 7))
        NSColor(calibratedRed: 0.10, green: 0.95, blue: 1.0, alpha: 0.98).setStroke()
        cross.lineWidth = 1.6
        cross.stroke()

        let dot = NSBezierPath(ovalIn: NSRect(x: cx - 2.5, y: cy - 2.5, width: 5, height: 5))
        NSColor(calibratedRed: 1.0, green: 0.85, blue: 0.15, alpha: 0.98).setFill()
        dot.fill()

        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 9.5, weight: .semibold),
            .foregroundColor: NSColor(white: 0.92, alpha: 0.95),
        ]
        ("\(title) EAR:\(ear)" as NSString).draw(at: NSPoint(x: rect.minX + 4, y: rect.maxY + 2), withAttributes: attrs)
    }

    private func drawEyeLockHUD(centerX: CGFloat, topY: CGFloat) {
        let hudW: CGFloat = 336
        let hudH: CGFloat = 104
        let hudRect = NSRect(x: centerX - hudW * 0.5, y: topY, width: hudW, height: hudH)

        let bg = NSBezierPath(roundedRect: hudRect, xRadius: 14, yRadius: 14)
        NSColor(calibratedWhite: 0.05, alpha: 0.86).setFill()
        bg.fill()

        let borderColor = eyeLocked
            ? NSColor(calibratedRed: 0.16, green: 0.96, blue: 0.56, alpha: 0.9)
            : NSColor(calibratedRed: 1.0, green: 0.42, blue: 0.25, alpha: 0.9)
        borderColor.setStroke()
        bg.lineWidth = 1.8
        bg.stroke()

        let statusText = eyeLocked
            ? "● HD 720p: НОС (\(noseX),\(noseY)) + ГЛАЗА ЗАФИКСИРОВАНЫ"
            : "○ ПОИСК ЛИЦА — СМОТРИТЕ В КАМЕРУ"
        let headerAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 10.5, weight: .bold),
            .foregroundColor: borderColor,
        ]
        let hSize = (statusText as NSString).size(withAttributes: headerAttrs)
        (statusText as NSString).draw(
            at: NSPoint(x: centerX - hSize.width * 0.5, y: topY + 8),
            withAttributes: headerAttrs
        )

        let eyeW: CGFloat = 116
        let eyeH: CGFloat = 56
        let leftRect = NSRect(x: hudRect.minX + 22, y: topY + 28, width: eyeW, height: eyeH)
        let rightRect = NSRect(x: hudRect.maxX - 22 - eyeW, y: topY + 28, width: eyeW, height: eyeH)
        drawEyeCrop(rect: leftRect, crop: leftEyeU4, pupX: pupLX, pupY: pupLY, ear: earL, title: "L")
        drawEyeCrop(rect: rightRect, crop: rightEyeU4, pupX: pupRX, pupY: pupRY, ear: earR, title: "R")

        // Tiny nose-pointer dot in center between the two eyes
        let nBox = NSRect(x: centerX - 18, y: topY + 38, width: 36, height: 36)
        let nBg = NSBezierPath(ovalIn: nBox)
        NSColor(calibratedWhite: 0.16, alpha: 0.9).setFill()
        nBg.fill()
        NSColor(calibratedRed: 0.15, green: 0.88, blue: 1.0, alpha: 0.7).setStroke()
        nBg.lineWidth = 1.2
        nBg.stroke()

        let nx = nBox.midX + max(-13, min(13, CGFloat(noseX) * 0.12))
        let ny = nBox.midY + max(-13, min(13, CGFloat(noseY) * 0.12))
        let nDot = NSBezierPath(ovalIn: NSRect(x: nx - 3.5, y: ny - 3.5, width: 7, height: 7))
        NSColor(calibratedRed: 0.18, green: 1.0, blue: 0.65, alpha: 1.0).setFill()
        nDot.fill()
    }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.clear.setFill()
        dirtyRect.fill()

        let w = bounds.width
        let h = bounds.height

        if mode == "intro" {
            NSColor(calibratedWhite: 0.02, alpha: 0.36).setFill()
            bounds.fill()

            let cardW: CGFloat = 720
            let cardH: CGFloat = 370
            let cardRect = NSRect(x: (w - cardW) * 0.5, y: (h - cardH) * 0.5, width: cardW, height: cardH)
            let card = NSBezierPath(roundedRect: cardRect, xRadius: 22, yRadius: 22)
            NSColor(calibratedWhite: 0.06, alpha: 0.93).setFill()
            card.fill()
            NSColor(calibratedRed: 0.14, green: 0.92, blue: 1.0, alpha: 0.92).setStroke()
            card.lineWidth = 2.5
            card.stroke()

            let titleStr = "TERNARA-GAZE v4 • НОС + ГЛАЗА + ФИКСАЦИЯ ОБЛАСТИ" as NSString
            let titleAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 22, weight: .bold),
                .foregroundColor: NSColor(calibratedRed: 0.20, green: 0.96, blue: 1.0, alpha: 1.0),
            ]
            let tSize = titleStr.size(withAttributes: titleAttrs)
            titleStr.draw(at: NSPoint(x: w * 0.5 - tSize.width * 0.5, y: cardRect.minY + 24), withAttributes: titleAttrs)

            let subAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 15, weight: .medium),
                .foregroundColor: NSColor(white: 0.94, alpha: 0.96),
            ]
            let lines = [
                "1. Направляйте взгляд и слегка поворачивайте нос на зелёную точку",
                "2. Когда кольцо станет ЗОЛОТЫМ — МОРГНИТЕ для перехода дальше",
                "3. После теста курсор будет МАГНИТНО ФИКСИРОВАТЬСЯ на области без дрожания",
            ]
            for (idx, line) in lines.enumerated() {
                let ns = line as NSString
                let sz = ns.size(withAttributes: subAttrs)
                ns.draw(
                    at: NSPoint(x: w * 0.5 - sz.width * 0.5, y: cardRect.minY + 62 + CGFloat(idx) * 24),
                    withAttributes: subAttrs
                )
            }

            drawEyeLockHUD(centerX: w * 0.5, topY: cardRect.minY + 152)

            let cdStr = "Старт через: \(introStep)" as NSString
            let cdAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedSystemFont(ofSize: 30, weight: .heavy),
                .foregroundColor: NSColor(calibratedRed: 0.25, green: 1.0, blue: 0.62, alpha: 1.0),
            ]
            let cdSize = cdStr.size(withAttributes: cdAttrs)
            cdStr.draw(
                at: NSPoint(x: w * 0.5 - cdSize.width * 0.5, y: cardRect.maxY - 62),
                withAttributes: cdAttrs
            )
            return
        }

        if mode == "calibrate" {
            let px = targetX * w
            let py = targetY * h
            let isBlinkWait = (calState == "BLINK")
            let pulse = 1.0 + 0.10 * sin(pulsePhase)

            let hudNearBottom = (targetY < 0.30)
            let bannerY: CGFloat = hudNearBottom ? (h - 204) : 34
            let hudY: CGFloat = hudNearBottom ? (h - 142) : 96

            let mainText = isBlinkWait
                ? "\(calLabel)  •  МОРГНИ ДЛЯ ПЕРЕХОДА!"
                : "\(calLabel)  •  СМОТРИ + НАПРАВЬ НОС (\(Int(calPct * 100))%)"
            let pillAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 18, weight: .bold),
                .foregroundColor: isBlinkWait
                    ? NSColor(calibratedRed: 1.0, green: 0.88, blue: 0.20, alpha: 1.0)
                    : NSColor.white,
            ]
            let txt = mainText as NSString
            let sz = txt.size(withAttributes: pillAttrs)
            let pillRect = NSRect(x: (w - sz.width) * 0.5 - 24, y: bannerY, width: sz.width + 48, height: 46)
            let pill = NSBezierPath(roundedRect: pillRect, xRadius: 23, yRadius: 23)
            NSColor(calibratedWhite: 0.05, alpha: 0.86).setFill()
            pill.fill()
            (isBlinkWait
                ? NSColor(calibratedRed: 1.0, green: 0.80, blue: 0.12, alpha: 0.98)
                : NSColor(calibratedRed: 0.18, green: 0.96, blue: 0.56, alpha: 0.95)
            ).setStroke()
            pill.lineWidth = 2.4
            pill.stroke()
            txt.draw(at: NSPoint(x: pillRect.minX + 24, y: pillRect.minY + 11), withAttributes: pillAttrs)

            drawEyeLockHUD(centerX: w * 0.5, topY: hudY)

            let outerR: CGFloat = (isBlinkWait ? 38.0 * pulse : 34.0)
            let outer = NSBezierPath(ovalIn: NSRect(x: px - outerR, y: py - outerR, width: outerR * 2, height: outerR * 2))
            if isBlinkWait {
                NSColor(calibratedRed: 1.0, green: 0.78, blue: 0.10, alpha: 0.30).setFill()
                outer.fill()
                NSColor(calibratedRed: 1.0, green: 0.84, blue: 0.15, alpha: 0.98).setStroke()
                outer.lineWidth = 4.5
                outer.stroke()
            } else {
                NSColor(calibratedRed: 0.10, green: 0.92, blue: 0.52, alpha: 0.22).setFill()
                outer.fill()
                NSColor(calibratedRed: 0.15, green: 0.96, blue: 0.56, alpha: 0.95).setStroke()
                outer.lineWidth = 3.0
                outer.stroke()
            }

            if calPct > 0.01 {
                let arc = NSBezierPath()
                arc.appendArc(
                    withCenter: NSPoint(x: px, y: py),
                    radius: outerR + 6,
                    startAngle: 90,
                    endAngle: 90 - 360 * min(1.0, calPct),
                    clockwise: true
                )
                (isBlinkWait
                    ? NSColor(calibratedRed: 1.0, green: 0.86, blue: 0.16, alpha: 1.0)
                    : NSColor(calibratedRed: 0.15, green: 0.98, blue: 1.0, alpha: 0.98)
                ).setStroke()
                arc.lineWidth = 5.5
                arc.lineCapStyle = .round
                arc.stroke()
            }

            let innerR: CGFloat = 6.5
            let inner = NSBezierPath(ovalIn: NSRect(x: px - innerR, y: py - innerR, width: innerR * 2, height: innerR * 2))
            (isBlinkWait ? NSColor(calibratedRed: 1.0, green: 0.90, blue: 0.25, alpha: 1.0) : NSColor.white).setFill()
            inner.fill()

            if isBlinkWait {
                let hint = "МОРГНИ!" as NSString
                let hAttrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: 15, weight: .heavy),
                    .foregroundColor: NSColor(calibratedRed: 1.0, green: 0.90, blue: 0.22, alpha: 1.0),
                ]
                let hSz = hint.size(withAttributes: hAttrs)
                let hx = max(14, min(w - hSz.width - 14, px - hSz.width * 0.5))
                let hy = (py > h * 0.80) ? (py - outerR - 30) : (py + outerR + 12)
                hint.draw(at: NSPoint(x: hx, y: hy), withAttributes: hAttrs)
            }
            return
        }

        // FLOATING SECOND CURSOR + MAGNETIC REGION LOCK MODE
        let b = screenBounds()
        let px: CGFloat
        let py: CGFloat
        if regionLocked, let sr = snappedRect {
            // Draw glowing VisionOS-style focus border around the locked macOS UI element!
            let localRect = NSRect(
                x: sr.origin.x - b.origin.x - 4,
                y: sr.origin.y - b.origin.y - 4,
                width: sr.size.width + 8,
                height: sr.size.height + 8
            )
            let focusBox = NSBezierPath(roundedRect: localRect, xRadius: 10, yRadius: 10)
            NSColor(calibratedRed: 0.14, green: 0.96, blue: 0.64, alpha: 0.12).setFill()
            focusBox.fill()
            NSColor(calibratedRed: 0.18, green: 0.98, blue: 0.68, alpha: 0.92).setStroke()
            focusBox.lineWidth = 2.6
            focusBox.stroke()

            px = localRect.midX
            py = localRect.midY
        } else {
            px = renderX * w
            py = renderY * h
        }

        // Compact top-right status badge
        let badgeText: String
        let badgeColor: NSColor
        if !eyeLocked {
            badgeText = "○ NO FACE"
            badgeColor = NSColor(calibratedRed: 1.0, green: 0.40, blue: 0.25, alpha: 0.95)
        } else if regionLocked && snappedRect != nil {
            badgeText = "● TARGET LOCKED"
            badgeColor = NSColor(calibratedRed: 0.18, green: 1.0, blue: 0.65, alpha: 0.98)
        } else if regionLocked {
            badgeText = "● ZONE LOCKED"
            badgeColor = NSColor(calibratedRed: 0.16, green: 0.96, blue: 0.62, alpha: 0.95)
        } else {
            badgeText = "◎ TRACKING (NOSE+EYE)"
            badgeColor = NSColor(calibratedRed: 0.15, green: 0.88, blue: 1.0, alpha: 0.95)
        }
        let bAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .bold),
            .foregroundColor: badgeColor,
        ]
        let bSz = (badgeText as NSString).size(withAttributes: bAttrs)
        let bRect = NSRect(x: w - bSz.width - 34, y: 34, width: bSz.width + 20, height: 24)
        let bPath = NSBezierPath(roundedRect: bRect, xRadius: 12, yRadius: 12)
        NSColor(calibratedWhite: 0.05, alpha: 0.76).setFill()
        bPath.fill()
        badgeColor.withAlphaComponent(0.8).setStroke()
        bPath.lineWidth = 1.3
        bPath.stroke()
        (badgeText as NSString).draw(at: NSPoint(x: bRect.minX + 10, y: bRect.minY + 4), withAttributes: bAttrs)

        // Click flash ripple
        if clickFlash > 0.02 {
            let rippleR: CGFloat = 18.0 + (1.0 - clickFlash) * 34.0
            let ripple = NSBezierPath(ovalIn: NSRect(x: px - rippleR, y: py - rippleR, width: rippleR * 2, height: rippleR * 2))
            let col = clickWasRight
                ? NSColor(calibratedRed: 1.0, green: 0.55, blue: 0.12, alpha: clickFlash * 0.85)
                : NSColor(calibratedRed: 0.15, green: 0.96, blue: 0.62, alpha: clickFlash * 0.85)
            col.setStroke()
            ripple.lineWidth = 3.5
            ripple.stroke()
        }

        // Reticle (turns crisp emerald-cyan when locked on a region, glowing cyan when gliding)
        let mainColor = regionLocked
            ? NSColor(calibratedRed: 0.18, green: 0.98, blue: 0.66, alpha: 0.96)
            : NSColor(calibratedRed: 0.15, green: 0.88, blue: 1.0, alpha: 0.90)

        let auraR: CGFloat = regionLocked ? 20.0 : 23.0
        let aura = NSBezierPath(ovalIn: NSRect(x: px - auraR, y: py - auraR, width: auraR * 2, height: auraR * 2))
        mainColor.withAlphaComponent(regionLocked ? 0.18 : 0.12).setFill()
        aura.fill()

        let baseR: CGFloat = regionLocked ? 12.5 : 14.5
        let ring = NSBezierPath(ovalIn: NSRect(x: px - baseR, y: py - baseR, width: baseR * 2, height: baseR * 2))
        mainColor.setStroke()
        ring.lineWidth = regionLocked ? 3.0 : 2.4
        ring.stroke()

        if holdPct > 0.02 {
            let arc = NSBezierPath()
            arc.appendArc(
                withCenter: NSPoint(x: px, y: py),
                radius: baseR + 5,
                startAngle: 90,
                endAngle: 90 - 360 * min(1.0, holdPct),
                clockwise: true
            )
            NSColor(calibratedRed: 1.0, green: 0.88, blue: 0.20, alpha: 0.98).setStroke()
            arc.lineWidth = 4.2
            arc.lineCapStyle = .round
            arc.stroke()
        }

        let dotR: CGFloat = 3.5
        let dot = NSBezierPath(ovalIn: NSRect(x: px - dotR, y: py - dotR, width: dotR * 2, height: dotR * 2))
        NSColor.white.setFill()
        dot.fill()
    }
}

final class OverlayController {
    let window: NSWindow
    let view: GazeOverlayView
    private var timer: Timer?
    private var lastLockedX: Int = -1
    private var lastLockedY: Int = -1
    private var cachedSnapPoint: CGPoint? = nil

    init() {
        let frame = NSScreen.main?.frame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        window = NSWindow(
            contentRect: frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.ignoresMouseEvents = true
        window.level = .screenSaver
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        view = GazeOverlayView(frame: NSRect(origin: .zero, size: frame.size))
        window.contentView = view
        window.orderFrontRegardless()

        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            let lerp: CGFloat = self.view.regionLocked ? 0.20 : 0.14
            self.view.renderX += (self.view.targetX - self.view.renderX) * lerp
            self.view.renderY += (self.view.targetY - self.view.renderY) * lerp
            self.view.pulsePhase += 0.16
            if self.view.clickFlash > 0.01 {
                self.view.clickFlash *= 0.84
            } else {
                self.view.clickFlash = 0.0
            }
            self.view.needsDisplay = true
        }
    }

    func updateEyeLock(
        locked: Bool,
        leftU4: [UInt8],
        rightU4: [UInt8],
        pupLX: Int,
        pupLY: Int,
        pupRX: Int,
        pupRY: Int,
        noseX: Int,
        noseY: Int,
        earL: Int,
        earR: Int
    ) {
        DispatchQueue.main.async {
            self.view.eyeLocked = locked
            if locked {
                self.view.leftEyeU4 = leftU4
                self.view.rightEyeU4 = rightU4
                self.view.pupLX = pupLX
                self.view.pupLY = pupLY
                self.view.pupRX = pupRX
                self.view.pupRY = pupRY
                self.view.noseX = noseX
                self.view.noseY = noseY
                self.view.earL = earL
                self.view.earR = earR
            }
        }
    }

    func updateIntro(step: String, subtitle: String) {
        DispatchQueue.main.async {
            self.view.mode = "intro"
            self.view.introStep = step
            if !subtitle.isEmpty {
                self.view.introSubtitle = subtitle
            }
        }
    }

    func updateGaze(xU16: Int, yU16: Int, holdPct: Int, locked: Bool) {
        let clampedX = max(0, min(65535, xU16))
        let clampedY = max(0, min(65535, yU16))
        var snapRect: CGRect? = nil
        var snapPt: CGPoint? = nil

        if locked {
            if abs(clampedX - lastLockedX) > 250 || abs(clampedY - lastLockedY) > 250 || cachedSnapPoint == nil {
                lastLockedX = clampedX
                lastLockedY = clampedY
                let pt = u16ToScreenPoint(xU16: clampedX, yU16: clampedY)
                if let r = queryMagneticTargetRect(screenPt: pt) {
                    snapRect = r
                    snapPt = CGPoint(x: r.midX, y: r.midY)
                }
                cachedSnapPoint = snapPt
                DispatchQueue.main.async {
                    self.view.snappedRect = snapRect
                }
            }
        } else {
            lastLockedX = -1
            lastLockedY = -1
            cachedSnapPoint = nil
            DispatchQueue.main.async {
                self.view.snappedRect = nil
            }
        }

        DispatchQueue.main.async {
            self.view.mode = "gaze"
            self.view.regionLocked = locked
            self.view.targetX = CGFloat(clampedX) / 65535.0
            self.view.targetY = CGFloat(clampedY) / 65535.0
            self.view.holdPct = CGFloat(max(0, min(100, holdPct))) / 100.0
        }
    }

    func clickPointFor(xU16: Int, yU16: Int) -> CGPoint {
        if let sp = cachedSnapPoint {
            return sp
        }
        return u16ToScreenPoint(xU16: xU16, yU16: yU16)
    }

    func updateCalTarget(xU16: Int, yU16: Int, pct: Int, state: String, label: String) {
        DispatchQueue.main.async {
            self.view.mode = "calibrate"
            self.view.targetX = CGFloat(max(0, min(65535, xU16))) / 65535.0
            self.view.targetY = CGFloat(max(0, min(65535, yU16))) / 65535.0
            self.view.calPct = CGFloat(max(0, min(100, pct))) / 100.0
            self.view.calState = state
            if !label.isEmpty {
                self.view.calLabel = label
            }
        }
    }

    func triggerClickFlash(right: Bool) {
        DispatchQueue.main.async {
            self.view.clickWasRight = right
            self.view.clickFlash = 1.0
        }
    }
}

// MARK: - Rigid HD Eye ROI, Sub-Pixel Pupil Geometry, EAR & 3D Nose Pointer

func cropEyeU4(
    yBase: UnsafePointer<UInt8>,
    width: Int,
    height: Int,
    bytesPerRow: Int,
    boxNorm: CGRect
) -> [UInt8] {
    let x0 = max(0.0, min(1.0, Double(boxNorm.minX)))
    let x1 = max(x0 + 0.01, min(1.0, Double(boxNorm.maxX)))
    let yTop = max(0.0, min(1.0, 1.0 - Double(boxNorm.maxY)))
    let yBot = max(yTop + 0.01, min(1.0, 1.0 - Double(boxNorm.minY)))

    var out = [UInt8](repeating: 0, count: EYE_W * EYE_H)
    for r in 0..<EYE_H {
        let fy = yTop + (Double(r) + 0.5) / Double(EYE_H) * (yBot - yTop)
        let py = max(0, min(height - 1, Int(fy * Double(height))))
        let rowPtr = yBase.advanced(by: py * bytesPerRow)
        for c in 0..<EYE_W {
            let fx = x0 + (Double(c) + 0.5) / Double(EYE_W) * (x1 - x0)
            let px = max(0, min(width - 1, Int(fx * Double(width))))
            out[r * EYE_W + c] = rowPtr[px] >> 4
        }
    }
    return out
}

func eyeBoxInImage(faceBox: CGRect, region: VNFaceLandmarkRegion2D) -> CGRect {
    let pts = region.normalizedPoints
    var minX: CGFloat = 1.0
    var maxX: CGFloat = 0.0
    var minY: CGFloat = 1.0
    var maxY: CGFloat = 0.0
    for p in pts {
        let ix = faceBox.origin.x + p.x * faceBox.size.width
        let iy = faceBox.origin.y + p.y * faceBox.size.height
        minX = min(minX, ix)
        maxX = max(maxX, ix)
        minY = min(minY, iy)
        maxY = max(maxY, iy)
    }
    let w = max(0.02, maxX - minX)
    let cx = (minX + maxX) * 0.5
    let cy = (minY + maxY) * 0.5
    let boxW = w * 1.55
    let boxH = boxW * 0.52
    return CGRect(x: cx - boxW * 0.5, y: cy - boxH * 0.5, width: boxW, height: boxH)
}

func rigidEyePupilAndEAR(
    eye: VNFaceLandmarkRegion2D,
    pupil: VNFaceLandmarkRegion2D?
) -> (px: Double, py: Double, ear: Int, center: CGPoint, width: CGFloat) {
    let pts = eye.normalizedPoints
    guard !pts.isEmpty else { return (0, 0, 0, CGPoint(x: 0.5, y: 0.5), 0.1) }

    var leftCorner = pts[0]
    var rightCorner = pts[0]
    var minY: CGFloat = 1.0
    var maxY: CGFloat = 0.0
    for pt in pts {
        if pt.x < leftCorner.x { leftCorner = pt }
        if pt.x > rightCorner.x { rightCorner = pt }
        minY = min(minY, pt.y)
        maxY = max(maxY, pt.y)
    }

    let w = max(0.008, rightCorner.x - leftCorner.x)
    let h = max(0.001, maxY - minY)
    let ear = max(0, min(500, Int((h / w) * 1000.0)))

    let cornerMidX = (leftCorner.x + rightCorner.x) * 0.5
    let cornerMidY = (leftCorner.y + rightCorner.y) * 0.5
    let center = CGPoint(x: cornerMidX, y: cornerMidY)

    guard let p = pupil?.normalizedPoints.first else {
        return (0, 0, ear, center, w)
    }
    let relX = -Double((p.x - cornerMidX) / w) * 320.0
    let relY = -Double((p.y - cornerMidY) / w) * 520.0
    return (relX, relY, ear, center, w)
}

func noseTipAndPose(
    faceBox: CGRect,
    leftEyeCenter: CGPoint,
    rightEyeCenter: CGPoint,
    nose: VNFaceLandmarkRegion2D?,
    noseCrest: VNFaceLandmarkRegion2D?
) -> (noseX: Double, noseY: Double, yaw: Double, pitch: Double) {
    let eyeMidX = (leftEyeCenter.x + rightEyeCenter.x) * 0.5
    let eyeMidY = (leftEyeCenter.y + rightEyeCenter.y) * 0.5
    let iod = max(0.05, abs(rightEyeCenter.x - leftEyeCenter.x))

    // Use bottom-most point of noseCrest (the true 3D nose tip!) or centroid of nose
    var tipLocal = CGPoint(x: eyeMidX, y: eyeMidY - 0.35 * iod)
    if let crest = noseCrest?.normalizedPoints, let last = crest.last {
        tipLocal = last
    } else if let nPts = nose?.normalizedPoints, !nPts.isEmpty {
        var nx: CGFloat = 0.0
        var ny: CGFloat = 0.0
        for p in nPts { nx += p.x; ny += p.y }
        tipLocal = CGPoint(x: nx / CGFloat(nPts.count), y: ny / CGFloat(nPts.count))
    }

    // Full image-space nose tip coordinate (rigid skull pointer!)
    let imgX = Double(faceBox.origin.x + tipLocal.x * faceBox.size.width)
    let imgY = 1.0 - Double(faceBox.origin.y + tipLocal.y * faceBox.size.height)
    let noseX = (0.5 - imgX) * 512.0   // mirrored so pointing right is positive
    let noseY = (imgY - 0.5) * 512.0   // pointing down is positive

    // 3D nose-tip vector relative to inter-ocular midpoint (pure head rotation!)
    let yaw = -Double((tipLocal.x - eyeMidX) / iod) * 380.0
    let pitch = (Double((eyeMidY - tipLocal.y) / iod) - 0.42) * 380.0
    return (noseX, noseY, yaw, pitch)
}

func packNibblesHex(left: [UInt8], right: [UInt8]) -> String {
    let hexChars = Array("0123456789abcdef".utf8)
    var all = [UInt8]()
    all.reserveCapacity(1024)
    all.append(contentsOf: left)
    all.append(contentsOf: right)
    var out = [UInt8]()
    out.reserveCapacity(1024)
    for i in stride(from: 0, to: 1024, by: 2) {
        let byte = ((all[i] & 0x0F) << 4) | (all[i + 1] & 0x0F)
        out.append(hexChars[Int(byte >> 4)])
        out.append(hexChars[Int(byte & 0x0F)])
    }
    return String(bytes: out, encoding: .ascii) ?? ""
}

final class GazeCaptureDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let startTime = DispatchTime.now().uptimeNanoseconds
    var rawFrames = 0
    var eyeFrames = 0
    let diagMode: Bool
    weak var overlay: OverlayController?

    // Gentle pre-EMA on raw landmarks to kill 1-frame Vision quantization noise before Python
    private var hasEma = false
    private var sNoseX = 0.0, sNoseY = 0.0, sYaw = 0.0, sPitch = 0.0
    private var sPupLX = 0.0, sPupLY = 0.0, sPupRX = 0.0, sPupRY = 0.0

    init(diagMode: Bool, overlay: OverlayController?) {
        self.diagMode = diagMode
        self.overlay = overlay
    }

    private func clampI8(_ v: Double) -> Int {
        return max(-128, min(127, Int(v.rounded())))
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        rawFrames += 1
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let req = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
        do {
            try handler.perform([req])
        } catch {
            if diagMode { fputs("vision_error=\(error)\n", stderr) }
            return
        }
        guard let face = req.results?.first,
              let lm = face.landmarks,
              let leftEye = lm.leftEye,
              let rightEye = lm.rightEye else {
            overlay?.updateEyeLock(
                locked: false, leftU4: [], rightU4: [],
                pupLX: 0, pupLY: 0, pupRX: 0, pupRY: 0, noseX: 0, noseY: 0, earL: 0, earR: 0
            )
            return
        }

        let leftBox = eyeBoxInImage(faceBox: face.boundingBox, region: leftEye)
        let rightBox = eyeBoxInImage(faceBox: face.boundingBox, region: rightEye)
        let lGeom = rigidEyePupilAndEAR(eye: leftEye, pupil: lm.leftPupil)
        let rGeom = rigidEyePupilAndEAR(eye: rightEye, pupil: lm.rightPupil)
        let nPose = noseTipAndPose(
            faceBox: face.boundingBox,
            leftEyeCenter: lGeom.center,
            rightEyeCenter: rGeom.center,
            nose: lm.nose,
            noseCrest: lm.noseCrest
        )

        if !hasEma {
            sNoseX = nPose.noseX; sNoseY = nPose.noseY
            sYaw = nPose.yaw; sPitch = nPose.pitch
            sPupLX = lGeom.px; sPupLY = lGeom.py
            sPupRX = rGeom.px; sPupRY = rGeom.py
            hasEma = true
        } else {
            let aNose = 0.45  // crisp nose tracking with zero micro-chatter
            let aPup = 0.35   // calmer pupil refinement
            sNoseX += (nPose.noseX - sNoseX) * aNose
            sNoseY += (nPose.noseY - sNoseY) * aNose
            sYaw += (nPose.yaw - sYaw) * aNose
            sPitch += (nPose.pitch - sPitch) * aNose
            sPupLX += (lGeom.px - sPupLX) * aPup
            sPupLY += (lGeom.py - sPupLY) * aPup
            sPupRX += (rGeom.px - sPupRX) * aPup
            sPupRY += (rGeom.py - sPupRY) * aPup
        }

        let noseXInt = clampI8(sNoseX)
        let noseYInt = clampI8(sNoseY)
        let yawInt = clampI8(sYaw)
        let pitchInt = clampI8(sPitch)
        let pupLXInt = clampI8(sPupLX)
        let pupLYInt = clampI8(sPupLY)
        let pupRXInt = clampI8(sPupRX)
        let pupRYInt = clampI8(sPupRY)

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

        guard let basePtr = CVPixelBufferGetBaseAddressOfPlane(pb, 0) else { return }
        let yPtr = basePtr.assumingMemoryBound(to: UInt8.self)
        let w = CVPixelBufferGetWidthOfPlane(pb, 0)
        let h = CVPixelBufferGetHeightOfPlane(pb, 0)
        let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)

        let leftU4 = cropEyeU4(yBase: yPtr, width: w, height: h, bytesPerRow: bpr, boxNorm: leftBox)
        let rightU4 = cropEyeU4(yBase: yPtr, width: w, height: h, bytesPerRow: bpr, boxNorm: rightBox)
        let hexStr = packNibblesHex(left: leftU4, right: rightU4)

        overlay?.updateEyeLock(
            locked: true,
            leftU4: leftU4,
            rightU4: rightU4,
            pupLX: pupLXInt,
            pupLY: pupLYInt,
            pupRX: pupRXInt,
            pupRY: pupRYInt,
            noseX: noseXInt,
            noseY: noseYInt,
            earL: lGeom.ear,
            earR: rGeom.ear
        )

        let nowMs = Int((DispatchTime.now().uptimeNanoseconds - startTime) / 1_000_000)
        let (mx, my) = currentMouseU16()
        eyeFrames += 1

        // Emit nose_x/nose_y (also aliased to head_x/head_y for clean backwards compatibility)
        let line = "{\"seq\":\(eyeFrames),\"ts_ms\":\(nowMs),\"head_x\":\(noseXInt),\"head_y\":\(noseYInt),\"nose_x\":\(noseXInt),\"nose_y\":\(noseYInt),\"yaw\":\(yawInt),\"pitch\":\(pitchInt),\"pup_lx\":\(pupLXInt),\"pup_ly\":\(pupLYInt),\"pup_rx\":\(pupRXInt),\"pup_ry\":\(pupRYInt),\"ear_l\":\(lGeom.ear),\"ear_r\":\(rGeom.ear),\"mx_u16\":\(mx),\"my_u16\":\(my),\"eyes_hex\":\"\(hexStr)\"}\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardOutput.write(data)
        }
    }
}

// MARK: - Main Entry

let args = CommandLine.arguments
if args.contains("--self-test") {
    let b = screenBounds()
    let ax = AXIsProcessTrusted()
    let camAuth = AVCaptureDevice.authorizationStatus(for: .video).rawValue
    print("{\"status\":\"ok\",\"screen_w\":\(Int(b.width)),\"screen_h\":\(Int(b.height)),\"camera_mode\":\"hd1280x720\",\"eye_roi\":\"32x16x4bit\",\"overlay\":\"virtual_second_cursor_v4_nose_region_lock\",\"accessibility_trusted\":\(ax),\"camera_auth\":\(camAuth)}")
    exit(0)
}

let diagMode = args.contains("--diag")
let overlayEnabled = args.contains("--overlay") || args.contains("--live")
let clicksEnabled = args.contains("--live")

let app = NSApplication.shared
app.setActivationPolicy(.accessory)

var overlay: OverlayController? = nil
if overlayEnabled {
    overlay = OverlayController()
    DispatchQueue.global(qos: .userInteractive).async {
        while let line = readLine() {
            let parts = line.split(separator: " ", maxSplits: 5)
            guard let cmd = parts.first else { continue }
            if cmd == "I", parts.count >= 2 {
                let step = String(parts[1])
                let sub = parts.count >= 3 ? parts[2...].joined(separator: " ") : ""
                overlay?.updateIntro(step: step, subtitle: sub)
            } else if cmd == "G", parts.count >= 3, let x = Int(parts[1]), let y = Int(parts[2]) {
                let pct = parts.count >= 4 ? (Int(parts[3]) ?? 0) : 0
                let locked = parts.count >= 5 ? ((Int(parts[4]) ?? 0) != 0) : false
                overlay?.updateGaze(xU16: x, yU16: y, holdPct: pct, locked: locked)
            } else if cmd == "T", parts.count >= 5, let x = Int(parts[1]), let y = Int(parts[2]), let pct = Int(parts[3]) {
                let state = String(parts[4])
                let lbl = parts.count >= 6 ? String(parts[5]) : ""
                overlay?.updateCalTarget(xU16: x, yU16: y, pct: pct, state: state, label: lbl)
            } else if cmd == "L", parts.count >= 3, let x = Int(parts[1]), let y = Int(parts[2]) {
                overlay?.triggerClickFlash(right: false)
                if clicksEnabled, let pt = overlay?.clickPointFor(xU16: x, yU16: y) {
                    atomicClickAtPoint(target: pt, right: false)
                }
            } else if cmd == "R", parts.count >= 3, let x = Int(parts[1]), let y = Int(parts[2]) {
                overlay?.triggerClickFlash(right: true)
                if clicksEnabled, let pt = overlay?.clickPointFor(xU16: x, yU16: y) {
                    atomicClickAtPoint(target: pt, right: true)
                }
            } else if cmd == "Q" {
                exit(0)
            }
        }
    }
}

let session = AVCaptureSession()
if session.canSetSessionPreset(.hd1280x720) {
    session.sessionPreset = .hd1280x720
} else {
    session.sessionPreset = .vga640x480
}

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera, .external],
    mediaType: .video,
    position: .unspecified
)
let device = discovery.devices.first(where: { $0.localizedName.contains("FaceTime") })
    ?? AVCaptureDevice.default(for: .video)

guard let cam = device else {
    fputs("ERROR: No camera device found.\n", stderr)
    exit(2)
}
guard let input = try? AVCaptureDeviceInput(device: cam), session.canAddInput(input) else {
    fputs("ERROR: Cannot open camera input.\n", stderr)
    exit(3)
}
session.addInput(input)

let output = AVCaptureVideoDataOutput()
output.videoSettings = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
]
output.alwaysDiscardsLateVideoFrames = true

let delegate = GazeCaptureDelegate(diagMode: diagMode, overlay: overlay)
let queue = DispatchQueue(label: "ternara.gaze.capture")
output.setSampleBufferDelegate(delegate, queue: queue)
guard session.canAddOutput(output) else {
    fputs("ERROR: Cannot add video output.\n", stderr)
    exit(4)
}
session.addOutput(output)
session.startRunning()

if diagMode {
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 4.0))
    session.stopRunning()
    fputs("diag summary raw_frames=\(delegate.rawFrames) eye_frames=\(delegate.eyeFrames)\n", stderr)
    exit(0)
}
app.run()
