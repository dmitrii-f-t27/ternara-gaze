import Foundation
import AVFoundation
import Vision
import CoreGraphics
import ApplicationServices

// Ternara-Gaze Native macOS Camera + Eye-ROI + CoreGraphics Mouse Bridge
// - Zero OpenCV / MediaPipe dependencies
// - Extracts ONLY dual 32x16 4-bit grayscale eye crops (512 bytes) from Y-plane luminance
// - Never writes raw video to disk
// - Accepts M <x_u16> <y_u16>, L (left click), R (right click) on stdin when --live is passed

let EYE_W = 32
let EYE_H = 16

func screenBounds() -> CGRect {
    return CGDisplayBounds(CGMainDisplayID())
}

func currentMouseU16() -> (Int, Int) {
    let bounds = screenBounds()
    let loc = CGEvent(source: nil)?.location ?? CGPoint(x: bounds.midX, y: bounds.midY)
    let nx = max(0.0, min(1.0, (loc.x - bounds.origin.x) / max(1.0, bounds.width)))
    let ny = max(0.0, min(1.0, (loc.y - bounds.origin.y) / max(1.0, bounds.height)))
    return (Int(nx * 65535.0), Int(ny * 65535.0))
}

func moveMouseU16(xU16: Int, yU16: Int) {
    let bounds = screenBounds()
    let x = bounds.origin.x + (Double(max(0, min(65535, xU16))) / 65535.0) * bounds.width
    let y = bounds.origin.y + (Double(max(0, min(65535, yU16))) / 65535.0) * bounds.height
    let pt = CGPoint(x: x, y: y)
    CGWarpMouseCursorPosition(pt)
    if let ev = CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: pt, mouseButton: .left) {
        ev.post(tap: .cghidEventTap)
    }
}

func clickMouse(right: Bool) {
    let loc = CGEvent(source: nil)?.location ?? CGPoint(x: 100, y: 100)
    let downType: CGEventType = right ? .rightMouseDown : .leftMouseDown
    let upType: CGEventType = right ? .rightMouseUp : .leftMouseUp
    let button: CGMouseButton = right ? .right : .left
    if let down = CGEvent(mouseEventSource: nil, mouseType: downType, mouseCursorPosition: loc, mouseButton: button),
       let up = CGEvent(mouseEventSource: nil, mouseType: upType, mouseCursorPosition: loc, mouseButton: button) {
        down.post(tap: .cghidEventTap)
        usleep(12_000)
        up.post(tap: .cghidEventTap)
    }
}

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
            if diagMode && (rawFrames == 1 || rawFrames % 15 == 0) {
                fputs("diag raw_frame=\(rawFrames) faces=\(req.results?.count ?? 0)\n", stderr)
            }
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

let args = CommandLine.arguments
if args.contains("--self-test") {
    let b = screenBounds()
    let ax = AXIsProcessTrusted()
    let camAuth = AVCaptureDevice.authorizationStatus(for: .video).rawValue
    print("{\"status\":\"ok\",\"screen_w\":\(Int(b.width)),\"screen_h\":\(Int(b.height)),\"eye_roi\":\"32x16x4bit\",\"accessibility_trusted\":\(ax),\"camera_auth\":\(camAuth)}")
    exit(0)
}

let diagMode = args.contains("--diag")
let liveMouse = args.contains("--live")
if liveMouse {
    DispatchQueue.global(qos: .userInteractive).async {
        while let line = readLine() {
            let parts = line.split(separator: " ")
            guard let cmd = parts.first else { continue }
            if cmd == "M", parts.count == 3, let x = Int(parts[1]), let y = Int(parts[2]) {
                moveMouseU16(xU16: x, yU16: y)
            } else if cmd == "L" {
                clickMouse(right: false)
            } else if cmd == "R" {
                clickMouse(right: true)
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
// Prefer built-in FaceTime HD camera over external BoardCam iPhones
let device = discovery.devices.first(where: { $0.localizedName.contains("FaceTime") })
    ?? AVCaptureDevice.default(for: .video)

guard let cam = device else {
    fputs("ERROR: No camera device found. Grant Camera permission to Terminal/Orca.\n", stderr)
    exit(2)
}
if diagMode {
    fputs("diag camera=\(cam.localizedName) auth=\(AVCaptureDevice.authorizationStatus(for: .video).rawValue) ax=\(AXIsProcessTrusted())\n", stderr)
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
RunLoop.main.run()
