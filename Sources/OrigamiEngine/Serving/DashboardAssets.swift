import CoreGraphics
import Foundation
import ImageIO

/// The dashboard's files besides the page, which the front and the model's server both serve without the key (they
/// hold no data): its icon, and what a phone or tablet needs to add the page to its home screen as an app.
extension DashboardPage {
    /// The page's Content-Security-Policy: nothing from elsewhere, the one inline script by its hash, `data:` images
    /// (the QR code under Access) and its own manifest.
    package static var csp: String {
        "default-src 'none'; style-src 'unsafe-inline'; script-src \(scriptHash); connect-src 'self'; img-src 'self' data:; manifest-src 'self'; "
            + "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    }

    /// The file served at `path` and its content type, or nil for anything else.
    public static func asset(_ path: String) -> (type: String, body: Data)? {
        switch path {
        case "/favicon.svg", "/favicon.ico": return ("image/svg+xml", Data(icon.utf8))
        case "/manifest.webmanifest": return ("application/manifest+json", manifest)
        // iOS asks for the precomposed name too, whatever the page says.
        case "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png": return ("image/png", touchIcon)
        case "/icon-192.png": return ("image/png", icon192)
        case "/icon-512.png": return ("image/png", icon512)
        default: return nil
        }
    }

    /// The web app manifest: relative URLs, so it also works behind a proxy path; colors from the page's dark tokens.
    static let manifest = Data(
        #"""
        {"name": "Origami", "short_name": "Origami", "start_url": "./dashboard", "scope": "./", "display": "standalone",
         "background_color": "#000000", "theme_color": "#000000",
         "icons": [{"src": "icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
                   {"src": "icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
                   {"src": "icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}]}
        """#.utf8)

    static let touchIcon = png(180), icon192 = png(192), icon512 = png(512)

    /// `icon`'s crane on a black square filling the canvas, without transparency (iOS rounds the corners itself), at
    /// three quarters of the canvas: inside the middle 80% that a maskable icon's crop keeps.
    static func png(_ size: Int) -> Data {
        // Each facet of `icon`'s SVG (its 32-unit box): gray level, then corners.
        let facets: [(CGFloat, [(CGFloat, CGFloat)])] = [
            (0.6, [(13, 19), (19, 3), (22, 19)]), (1, [(15, 20), (27, 5), (23, 20)]), (1, [(8, 22), (6, 8), (14, 20)]),
            (0.8, [(6, 8), (2, 14), (8, 13)]), (0.8, [(21, 21), (30, 15), (22, 16)]), (1, [(7, 22), (16, 18), (25, 22)]),
            (2 / 3, [(7, 22), (25, 22), (16, 27)]),
        ]
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return Data() }
        let side = CGFloat(size)
        let scale = side / 32 * 0.75
        context.setFillColor(CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        // The crane's box (2…30, 3…27) centred; CoreGraphics counts y upwards.
        let point = { (p: (CGFloat, CGFloat)) in CGPoint(x: side / 2 + (p.0 - 16) * scale, y: side / 2 - (p.1 - 15) * scale) }
        for (level, corners) in facets {
            context.setFillColor(CGColor(srgbRed: level, green: level, blue: level, alpha: 1))
            context.addLines(between: corners.map(point))
            context.closePath()
            context.fillPath()
        }
        let data = NSMutableData()
        guard let image = context.makeImage(), let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)
        else { return Data() }
        CGImageDestinationAddImage(destination, image, nil)
        return CGImageDestinationFinalize(destination) ? data as Data : Data()
    }
}
