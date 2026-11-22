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

func atomicClickAtU16(xU16: Int, yU16: Int, right: Bool) {
    let b = screenBounds()
    let prevPt = CGEvent(source: nil)?.location ?? CGPoint(x: b.midX, y: b.midY)
    let nx = Double(max(0, min(65535, xU16))) / 65535.0
    let ny = Double(max(0, min(65535, yU16))) / 65535.0
    let target = CGPoint(x: b.origin.x + nx * b.size.width, y: b.origin.y + ny * b.size.height)

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

// MARK: - Smooth 60 FPS Floating Second Cursor + Live Eye-Lock HUD + Intro + Blink-to-Advance

final class GazeOverlayView: NSView {
    var mode: String = "intro"   // "intro" | "calibrate" | "gaze"
    var targetX: CGFloat = 0.5
    var targetY: CGFloat = 0.5
    var renderX: CGFloat = 0.5
    var renderY: CGFloat = 0.5
    var holdPct: CGFloat = 0.0

    var calPct: CGFloat = 0.0
    var calState: String = "AIM" // "AIM" | "REC" | "BLINK"
    var calLabel: String = ""

    var introStep: String = "3"
    var introSubtitle: String = "Камера фиксирует область глаз в HD 720p"

    var clickFlash: CGFloat = 0.0
    var clickWasRight: Bool = false
    var pulsePhase: CGFloat = 0.0

    // Live Eye-Lock telemetry (purely in RAM, never saved)
    var eyeLocked: Bool = false
    var leftEyeU4: [UInt8] = [UInt8](repeating: 0, count: EYE_W * EYE_H)
    var rightEyeU4: [UInt8] = [UInt8](repeating: 0, count: EYE_W * EYE_H)
    var pupLX: Int = 0
    var pupLY: Int = 0
    var pupRX: Int = 0
    var pupRY: Int = 0
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
        // Draw 32x16 4-bit grayscale eye crop cleanly via CGImage
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
                // Unflip inside flipped view so the eye renders right-side up
                ctx.translateBy(x: rect.minX, y: rect.maxY)
                ctx.scaleBy(x: 1.0, y: -1.0)
                ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: rect.width, height: rect.height))
                ctx.restoreGState()
            }
        }

        // Green lock frame around eye
        let lockColor = eyeLocked
            ? NSColor(calibratedRed: 0.15, green: 0.96, blue: 0.55, alpha: 0.95)
            : NSColor(calibratedRed: 1.0, green: 0.35, blue: 0.25, alpha: 0.95)
        let border = NSBezierPath(roundedRect: rect, xRadius: 5, yRadius: 5)
        lockColor.setStroke()
        border.lineWidth = 2.0
        border.stroke()

        // Crosshair at detected pupil offset
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
        let hudW: CGFloat = 286
        let hudH: CGFloat = 96
        let hudRect = NSRect(x: centerX - hudW * 0.5, y: topY, width: hudW, height: hudH)

        let bg = NSBezierPath(roundedRect: hudRect, xRadius: 14, yRadius: 14)
        NSColor(calibratedWhite: 0.05, alpha: 0.84).setFill()
        bg.fill()

        let borderColor = eyeLocked
            ? NSColor(calibratedRed: 0.16, green: 0.96, blue: 0.56, alpha: 0.9)
            : NSColor(calibratedRed: 1.0, green: 0.42, blue: 0.25, alpha: 0.9)
        borderColor.setStroke()
        bg.lineWidth = 1.8
        bg.stroke()

        let statusText = eyeLocked
            ? "● HD КАМЕРА: ГЛАЗА ЗАФИКСИРОВАНЫ"
            : "○ ПОИСК ГЛАЗ — СМОТРИТЕ В КАМЕРУ"
        let headerAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 11.5, weight: .bold),
            .foregroundColor: borderColor,
        ]
        let hSize = (statusText as NSString).size(withAttributes: headerAttrs)
        (statusText as NSString).draw(
            at: NSPoint(x: centerX - hSize.width * 0.5, y: topY + 7),
            withAttributes: headerAttrs
        )

        let eyeW: CGFloat = 112
        let eyeH: CGFloat = 54
        let leftRect = NSRect(x: hudRect.minX + 20, y: topY + 26, width: eyeW, height: eyeH)
        let rightRect = NSRect(x: hudRect.maxX - 20 - eyeW, y: topY + 26, width: eyeW, height: eyeH)
        drawEyeCrop(rect: leftRect, crop: leftEyeU4, pupX: pupLX, pupY: pupLY, ear: earL, title: "L")
        drawEyeCrop(rect: rightRect, crop: rightEyeU4, pupX: pupRX, pupY: pupRY, ear: earR, title: "R")
    }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.clear.setFill()
        dirtyRect.fill()

        let w = bounds.width
        let h = bounds.height

        if mode == "intro" {
            // Subtle dim backdrop so the start card is unmistakable
            NSColor(calibratedWhite: 0.02, alpha: 0.36).setFill()
            bounds.fill()

            let cardW: CGFloat = 680
            let cardH: CGFloat = 350
            let cardRect = NSRect(x: (w - cardW) * 0.5, y: (h - cardH) * 0.5, width: cardW, height: cardH)
            let card = NSBezierPath(roundedRect: cardRect, xRadius: 22, yRadius: 22)
            NSColor(calibratedWhite: 0.06, alpha: 0.92).setFill()
            card.fill()
            NSColor(calibratedRed: 0.14, green: 0.92, blue: 1.0, alpha: 0.92).setStroke()
            card.lineWidth = 2.5
            card.stroke()

            let titleStr = "TERNARA-GAZE • ЗАПУСК ТЕСТА КАЛИБРОВКИ" as NSString
            let titleAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 23, weight: .bold),
                .foregroundColor: NSColor(calibratedRed: 0.20, green: 0.96, blue: 1.0, alpha: 1.0),
            ]
            let tSize = titleStr.size(withAttributes: titleAttrs)
            titleStr.draw(at: NSPoint(x: w * 0.5 - tSize.width * 0.5, y: cardRect.minY + 24), withAttributes: titleAttrs)

            let subAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 15, weight: .medium),
                .foregroundColor: NSColor(white: 0.94, alpha: 0.96),
            ]
            let lines = [
                "1. Смотрите на зелёную точку по углам экрана, пока кольцо заполняется",
                "2. Когда кольцо станет ЗОЛОТЫМ — МОРГНИТЕ для перехода к следующей точке",
                introSubtitle,
            ]
            for (idx, line) in lines.enumerated() {
                let ns = line as NSString
                let sz = ns.size(withAttributes: subAttrs)
                ns.draw(
                    at: NSPoint(x: w * 0.5 - sz.width * 0.5, y: cardRect.minY + 64 + CGFloat(idx) * 24),
                    withAttributes: subAttrs
                )
            }

            // Live HD Eye-Lock preview right inside the start card
            drawEyeLockHUD(centerX: w * 0.5, topY: cardRect.minY + 148)

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

            // Position banner + eye HUD away from top targets so nothing overlaps!
            let hudNearBottom = (targetY < 0.30)
            let bannerY: CGFloat = hudNearBottom ? (h - 196) : 34
            let hudY: CGFloat = hudNearBottom ? (h - 134) : 96

            // Banner pill
            let mainText: String
            if isBlinkWait {
                mainText = "\(calLabel)  •  МОРГНИ ДЛЯ ПЕРЕХОДА!"
            } else {
                mainText = "\(calLabel)  •  СМОТРИ НА ТОЧКУ (\(Int(calPct * 100))%)"
            }
            let pillAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 18, weight: .bold),
                .foregroundColor: isBlinkWait
                    ? NSColor(calibratedRed: 1.0, green: 0.88, blue: 0.20, alpha: 1.0)
                    : NSColor.white,
            ]
            let txt = mainText as NSString
            let sz = txt.size(withAttributes: pillAttrs)
            let pillRect = NSRect(
                x: (w - sz.width) * 0.5 - 24,
                y: bannerY,
                width: sz.width + 48,
                height: 46
            )
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

            // Live Eye-Lock HUD so user confirms MacBook camera stays locked on eyes
            drawEyeLockHUD(centerX: w * 0.5, topY: hudY)

            // Calibration target circle
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

            // Crisp center fixation point
            let innerR: CGFloat = 6.5
            let inner = NSBezierPath(ovalIn: NSRect(x: px - innerR, y: py - innerR, width: innerR * 2, height: innerR * 2))
            (isBlinkWait ? NSColor(calibratedRed: 1.0, green: 0.90, blue: 0.25, alpha: 1.0) : NSColor.white).setFill()
            inner.fill()

            // Local label right next to dot when waiting for blink
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

        // Floating second virtual cursor mode (never hijacks physical mouse!)
        let px = renderX * w
        let py = renderY * h

        // Compact top-right eye-lock status badge (non-intrusive)
        let badgeStr = (eyeLocked ? "● EYE LOCK HD" : "○ EYE LOST") as NSString
        let badgeColor = eyeLocked
            ? NSColor(calibratedRed: 0.16, green: 0.96, blue: 0.56, alpha: 0.95)
            : NSColor(calibratedRed: 1.0, green: 0.40, blue: 0.25, alpha: 0.95)
        let bAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .bold),
            .foregroundColor: badgeColor,
        ]
        let bSz = badgeStr.size(withAttributes: bAttrs)
        let bRect = NSRect(x: w - bSz.width - 34, y: 34, width: bSz.width + 20, height: 24)
        let bPath = NSBezierPath(roundedRect: bRect, xRadius: 12, yRadius: 12)
        NSColor(calibratedWhite: 0.05, alpha: 0.72).setFill()
        bPath.fill()
        badgeColor.withAlphaComponent(0.75).setStroke()
        bPath.lineWidth = 1.2
        bPath.stroke()
        badgeStr.draw(at: NSPoint(x: bRect.minX + 10, y: bRect.minY + 4), withAttributes: bAttrs)

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

        // Soft glowing aura
        let auraR: CGFloat = 22.0
        let aura = NSBezierPath(ovalIn: NSRect(x: px - auraR, y: py - auraR, width: auraR * 2, height: auraR * 2))
        NSColor(calibratedRed: 0.12, green: 0.84, blue: 1.0, alpha: 0.14).setFill()
        aura.fill()

        // Main smooth cyan ring
        let baseR: CGFloat = 14.0
        let ring = NSBezierPath(ovalIn: NSRect(x: px - baseR, y: py - baseR, width: baseR * 2, height: baseR * 2))
        NSColor(calibratedRed: 0.15, green: 0.88, blue: 1.0, alpha: 0.92).setStroke()
        ring.lineWidth = 2.6
        ring.stroke()

        // Deliberate blink-click progress arc
        if holdPct > 0.02 {
            let arc = NSBezierPath()
            arc.appendArc(
                withCenter: NSPoint(x: px, y: py),
                radius: baseR + 5,
                startAngle: 90,
                endAngle: 90 - 360 * min(1.0, holdPct),
                clockwise: true
            )
            NSColor(calibratedRed: 0.20, green: 1.0, blue: 0.60, alpha: 0.98).setStroke()
            arc.lineWidth = 4.0
            arc.lineCapStyle = .round
            arc.stroke()
        }

        // Crisp center dot
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
            // Critically-damped smooth 60 FPS glide (zero micro-jitter!)
            let lerp: CGFloat = 0.14
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

    func updateGaze(xU16: Int, yU16: Int, holdPct: Int) {
        DispatchQueue.main.async {
            self.view.mode = "gaze"
            self.view.targetX = CGFloat(max(0, min(65535, xU16))) / 65535.0
            self.view.targetY = CGFloat(max(0, min(65535, yU16))) / 65535.0
            self.view.holdPct = CGFloat(max(0, min(100, holdPct))) / 100.0
        }
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

// MARK: - Rigid HD Eye ROI, Sub-Pixel Pupil Geometry, EAR & 3D Head Pose

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
    // Keep crop aspect ratio fixed to rigid eye width so blinking never zooms the ROI!
    let boxW = w * 1.55
    let boxH = boxW * 0.52
    return CGRect(x: cx - boxW * 0.5, y: cy - boxH * 0.5, width: boxW, height: boxH)
}

func rigidEyePupilAndEAR(
    eye: VNFaceLandmarkRegion2D,
    pupil: VNFaceLandmarkRegion2D?
) -> (px: Int, py: Int, ear: Int, center: CGPoint, width: CGFloat) {
    let pts = eye.normalizedPoints
    guard !pts.isEmpty else { return (0, 0, 0, CGPoint(x: 0.5, y: 0.5), 0.1) }

    // Find rigid inner & outer canthus corners (minX and maxX points)
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

    // Rigid anchor midpoint on the bony eye-corner line (does NOT bounce when eyelids squint!)
    let cornerMidX = (leftCorner.x + rightCorner.x) * 0.5
    let cornerMidY = (leftCorner.y + rightCorner.y) * 0.5
    let center = CGPoint(x: cornerMidX, y: cornerMidY)

    guard let p = pupil?.normalizedPoints.first else {
        return (0, 0, ear, center, w)
    }
    // Divide BOTH X and Y by rigid eye width `w` (fixes vertical eyelid-height jitter!)
    let relX = -Double((p.x - cornerMidX) / w) * 320.0
    let relY = -Double((p.y - cornerMidY) / w) * 520.0
    let px = max(-128, min(127, Int(relX)))
    let py = max(-128, min(127, Int(relY)))
    return (px, py, ear, center, w)
}

func landmarkYawPitch(
    leftEyeCenter: CGPoint,
    rightEyeCenter: CGPoint,
    nose: VNFaceLandmarkRegion2D?
) -> (Int, Int) {
    let eyeMidX = (leftEyeCenter.x + rightEyeCenter.x) * 0.5
    let eyeMidY = (leftEyeCenter.y + rightEyeCenter.y) * 0.5
    let iod = max(0.05, abs(rightEyeCenter.x - leftEyeCenter.x))
    guard let nPts = nose?.normalizedPoints, !nPts.isEmpty else { return (0, 0) }
    var nx: CGFloat = 0.0
    var ny: CGFloat = 0.0
    for p in nPts {
        nx += p.x
        ny += p.y
    }
    nx /= CGFloat(nPts.count)
    ny /= CGFloat(nPts.count)

    // Horizontal nose offset from eye midpoint -> head yaw (mirrored)
    let yawVal = -Double((nx - eyeMidX) / iod) * 360.0
    // Vertical nose drop below eye line -> head pitch (neutral ~0.42*IOD)
    let pitchRatio = Double((eyeMidY - ny) / iod) - 0.42
    let pitchVal = pitchRatio * 340.0
    return (
        max(-128, min(127, Int(yawVal))),
        max(-128, min(127, Int(pitchVal)))
    )
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

    init(diagMode: Bool, overlay: OverlayController?) {
        self.diagMode = diagMode
        self.overlay = overlay
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
                pupLX: 0, pupLY: 0, pupRX: 0, pupRY: 0, earL: 0, earR: 0
            )
            return
        }

        let leftBox = eyeBoxInImage(faceBox: face.boundingBox, region: leftEye)
        let rightBox = eyeBoxInImage(faceBox: face.boundingBox, region: rightEye)
        let lGeom = rigidEyePupilAndEAR(eye: leftEye, pupil: lm.leftPupil)
        let rGeom = rigidEyePupilAndEAR(eye: rightEye, pupil: lm.rightPupil)
        let (yawInt, pitchInt) = landmarkYawPitch(
            leftEyeCenter: lGeom.center,
            rightEyeCenter: rGeom.center,
            nose: lm.nose ?? lm.noseCrest
        )

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
            pupLX: lGeom.px,
            pupLY: lGeom.py,
            pupRX: rGeom.px,
            pupRY: rGeom.py,
            earL: lGeom.ear,
            earR: rGeom.ear
        )

        let faceCx = Double(face.boundingBox.midX)
        let faceCy = 1.0 - Double(face.boundingBox.midY)
        let headX = max(-128, min(127, Int((0.5 - faceCx) * 256.0)))
        let headY = max(-128, min(127, Int((faceCy - 0.5) * 256.0)))

        let nowMs = Int((DispatchTime.now().uptimeNanoseconds - startTime) / 1_000_000)
        let (mx, my) = currentMouseU16()
        eyeFrames += 1

        let line = "{\"seq\":\(eyeFrames),\"ts_ms\":\(nowMs),\"head_x\":\(headX),\"head_y\":\(headY),\"yaw\":\(yawInt),\"pitch\":\(pitchInt),\"pup_lx\":\(lGeom.px),\"pup_ly\":\(lGeom.py),\"pup_rx\":\(rGeom.px),\"pup_ry\":\(rGeom.py),\"ear_l\":\(lGeom.ear),\"ear_r\":\(rGeom.ear),\"mx_u16\":\(mx),\"my_u16\":\(my),\"eyes_hex\":\"\(hexStr)\"}\n"
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
    print("{\"status\":\"ok\",\"screen_w\":\(Int(b.width)),\"screen_h\":\(Int(b.height)),\"camera_mode\":\"hd1280x720\",\"eye_roi\":\"32x16x4bit\",\"overlay\":\"virtual_second_cursor_v3_eyelock\",\"accessibility_trusted\":\(ax),\"camera_auth\":\(camAuth)}")
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
                overlay?.updateGaze(xU16: x, yU16: y, holdPct: pct)
            } else if cmd == "T", parts.count >= 5, let x = Int(parts[1]), let y = Int(parts[2]), let pct = Int(parts[3]) {
                let state = String(parts[4])
                let lbl = parts.count >= 6 ? String(parts[5]) : ""
                overlay?.updateCalTarget(xU16: x, yU16: y, pct: pct, state: state, label: lbl)
            } else if cmd == "L", parts.count >= 3, let x = Int(parts[1]), let y = Int(parts[2]) {
                overlay?.triggerClickFlash(right: false)
                if clicksEnabled {
                    atomicClickAtU16(xU16: x, yU16: y, right: false)
                }
            } else if cmd == "R", parts.count >= 3, let x = Int(parts[1]), let y = Int(parts[2]) {
                overlay?.triggerClickFlash(right: true)
                if clicksEnabled {
                    atomicClickAtU16(xU16: x, yU16: y, right: true)
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
