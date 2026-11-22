import Foundation
import AppKit
import AVFoundation
import Vision
import CoreGraphics
import ApplicationServices

// Ternara-Gaze Native macOS Camera + Floating Second Cursor Overlay + Atomic Computer-Use Click
// - Your real physical mouse/trackpad is NEVER hijacked or fought over.
// - A silky-smooth 60 FPS click-through virtual reticle follows your gaze.
// - On blink/wink click, posts an atomic CGEvent click at the reticle position and immediately
//   restores your physical mouse cursor to where you left it.
// - During 9-point calibration, displays glowing on-screen target dots at the exact screen points.

let EYE_W = 32
let EYE_H = 16

func screenBounds() -> CGRect {
    return CGDisplayBounds(CGMainDisplayID())
}

func u16ToCGPoint(xU16: Int, yU16: Int) -> CGPoint {
    let b = screenBounds()
    let nx = Double(max(0, min(65535, xU16))) / 65535.0
    let ny = Double(max(0, min(65535, yU16))) / 65535.0
    return CGPoint(x: b.origin.x + nx * b.width, y: b.origin.y + ny * b.height)
}

func currentMouseU16() -> (Int, Int) {
    let bounds = screenBounds()
    let loc = CGEvent(source: nil)?.location ?? CGPoint(x: bounds.midX, y: bounds.midY)
    let nx = max(0.0, min(1.0, (loc.x - bounds.origin.x) / max(1.0, bounds.width)))
    let ny = max(0.0, min(1.0, (loc.y - bounds.origin.y) / max(1.0, bounds.height)))
    return (Int(nx * 65535.0), Int(ny * 65535.0))
}

func atomicClickAtU16(xU16: Int, yU16: Int, right: Bool) {
    // Save user's real physical mouse position, click cleanly at gaze reticle, restore real mouse!
    let savedLoc = CGEvent(source: nil)?.location
    let pt = u16ToCGPoint(xU16: xU16, yU16: yU16)
    let downType: CGEventType = right ? .rightMouseDown : .leftMouseDown
    let upType: CGEventType = right ? .rightMouseUp : .leftMouseUp
    let button: CGMouseButton = right ? .right : .left

    if let move = CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: pt, mouseButton: .left),
       let down = CGEvent(mouseEventSource: nil, mouseType: downType, mouseCursorPosition: pt, mouseButton: button),
       let up = CGEvent(mouseEventSource: nil, mouseType: upType, mouseCursorPosition: pt, mouseButton: button) {
        move.post(tap: .cghidEventTap)
        usleep(4_000)
        down.post(tap: .cghidEventTap)
        usleep(14_000)
        up.post(tap: .cghidEventTap)
    }
    if let orig = savedLoc {
        usleep(6_000)
        CGWarpMouseCursorPosition(orig)
    }
}

// MARK: - Native Floating Overlay (Second Cursor + Calibration Targets)

final class GazeOverlayView: NSView {
    var mode: String = "idle" // "idle", "gaze", "calibrate"
    var targetX: CGFloat = 0.5
    var targetY: CGFloat = 0.5
    var renderX: CGFloat = 0.5
    var renderY: CGFloat = 0.5
    var holdPct: CGFloat = 0.0
    var calPct: CGFloat = 0.0
    var clickFlash: CGFloat = 0.0
    var clickWasRight: Bool = false

    override func draw(_ dirtyRect: NSRect) {
        NSColor.clear.set()
        dirtyRect.fill()

        guard mode != "idle" else { return }
        let w = bounds.width
        let h = bounds.height

        // AppKit origin is bottom-left; gaze Y=0 is top of screen
        let px = renderX * w
        let py = (1.0 - renderY) * h

        if mode == "calibrate" {
            let cx = targetX * w
            let cy = (1.0 - targetY) * h
            // Outer halo
            let outerR: CGFloat = 34
            let outerRect = NSRect(x: cx - outerR, y: cy - outerR, width: outerR * 2, height: outerR * 2)
            let outer = NSBezierPath(ovalIn: outerRect)
            NSColor(calibratedRed: 0.05, green: 0.85, blue: 0.65, alpha: 0.22).setFill()
            outer.fill()
            NSColor(calibratedRed: 0.10, green: 0.95, blue: 0.75, alpha: 0.95).setStroke()
            outer.lineWidth = 3.0
            outer.stroke()

            // Progress arc
            if calPct > 0.01 {
                let arc = NSBezierPath()
                arc.appendArc(
                    withCenter: NSPoint(x: cx, y: cy),
                    radius: outerR + 6,
                    startAngle: 90,
                    endAngle: 90 - 360 * calPct,
                    clockwise: true
                )
                NSColor(calibratedRed: 1.0, green: 0.82, blue: 0.15, alpha: 0.95).setStroke()
                arc.lineWidth = 5.0
                arc.lineCapStyle = .round
                arc.stroke()
            }

            // Inner bright fixation dot
            let dotR: CGFloat = 8
            let dot = NSBezierPath(ovalIn: NSRect(x: cx - dotR, y: cy - dotR, width: dotR * 2, height: dotR * 2))
            NSColor.white.setFill()
            dot.fill()
            return
        }

        // Virtual Second Cursor ("Ternara Gaze Reticle")
        let baseR: CGFloat = 18.0
        let haloR: CGFloat = baseR + 10.0 * clickFlash

        // Soft glow backdrop
        let glowRect = NSRect(x: px - haloR - 4, y: py - haloR - 4, width: (haloR + 4) * 2, height: (haloR + 4) * 2)
        let glow = NSBezierPath(ovalIn: glowRect)
        let ringColor: NSColor = clickFlash > 0.05
            ? (clickWasRight
                ? NSColor(calibratedRed: 1.0, green: 0.45, blue: 0.20, alpha: 0.95)
                : NSColor(calibratedRed: 0.15, green: 1.0, blue: 0.65, alpha: 0.95))
            : NSColor(calibratedRed: 0.10, green: 0.88, blue: 0.98, alpha: 0.90)

        ringColor.withAlphaComponent(0.16 + 0.22 * clickFlash).setFill()
        glow.fill()

        // Main crisp reticle ring
        let ring = NSBezierPath(ovalIn: NSRect(x: px - baseR, y: py - baseR, width: baseR * 2, height: baseR * 2))
        ringColor.setStroke()
        ring.lineWidth = 2.6
        ring.stroke()

        // Blink-click hold progress ring
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

        // Crisp center pointer dot
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

        // 60 FPS butter-smooth spring interpolation on the virtual cursor
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            let lerp: CGFloat = 0.24
            self.view.renderX += (self.view.targetX - self.view.renderX) * lerp
            self.view.renderY += (self.view.targetY - self.view.renderY) * lerp
            if self.view.clickFlash > 0.01 {
                self.view.clickFlash *= 0.84
            } else {
                self.view.clickFlash = 0.0
            }
            self.view.needsDisplay = true
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

    func updateCalTarget(xU16: Int, yU16: Int, pct: Int) {
        DispatchQueue.main.async {
            self.view.mode = "calibrate"
            self.view.targetX = CGFloat(max(0, min(65535, xU16))) / 65535.0
            self.view.targetY = CGFloat(max(0, min(65535, yU16))) / 65535.0
            self.view.calPct = CGFloat(max(0, min(100, pct))) / 100.0
        }
    }

    func triggerClickFlash(right: Bool) {
        DispatchQueue.main.async {
            self.view.clickWasRight = right
            self.view.clickFlash = 1.0
        }
    }
}

// MARK: - Eye ROI Extraction

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
    let h = max(0.015, maxY - minY)
    let cx = (minX + maxX) * 0.5
    let cy = (minY + maxY) * 0.5
    let boxW = w * 1.65
    let boxH = max(h * 2.2, boxW * 0.5)
    return CGRect(x: cx - boxW * 0.5, y: cy - boxH * 0.5, width: boxW, height: boxH)
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

    init(diagMode: Bool) {
        self.diagMode = diagMode
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
            return
        }

        let leftBox = eyeBoxInImage(faceBox: face.boundingBox, region: leftEye)
        let rightBox = eyeBoxInImage(faceBox: face.boundingBox, region: rightEye)

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

        let faceCx = Double(face.boundingBox.midX)
        let faceCy = 1.0 - Double(face.boundingBox.midY)
        let headX = max(-128, min(127, Int((0.5 - faceCx) * 256.0)))
        let headY = max(-128, min(127, Int((faceCy - 0.5) * 256.0)))
        let nowMs = Int((DispatchTime.now().uptimeNanoseconds - startTime) / 1_000_000)
        let (mx, my) = currentMouseU16()
        eyeFrames += 1

        let line = "{\"seq\":\(eyeFrames),\"ts_ms\":\(nowMs),\"head_x\":\(headX),\"head_y\":\(headY),\"mx_u16\":\(mx),\"my_u16\":\(my),\"eyes_hex\":\"\(hexStr)\"}\n"
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
    print("{\"status\":\"ok\",\"screen_w\":\(Int(b.width)),\"screen_h\":\(Int(b.height)),\"eye_roi\":\"32x16x4bit\",\"overlay\":\"virtual_second_cursor\",\"accessibility_trusted\":\(ax),\"camera_auth\":\(camAuth)}")
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
            let parts = line.split(separator: " ")
            guard let cmd = parts.first else { continue }
            if cmd == "G", parts.count >= 3, let x = Int(parts[1]), let y = Int(parts[2]) {
                let pct = parts.count >= 4 ? (Int(parts[3]) ?? 0) : 0
                overlay?.updateGaze(xU16: x, yU16: y, holdPct: pct)
            } else if cmd == "T", parts.count >= 4, let x = Int(parts[1]), let y = Int(parts[2]), let pct = Int(parts[3]) {
                overlay?.updateCalTarget(xU16: x, yU16: y, pct: pct)
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
session.sessionPreset = .vga640x480

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

let delegate = GazeCaptureDelegate(diagMode: diagMode)
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
