import { NextRequest, NextResponse } from "next/server";

// SSRF Protection: Private IP ranges and blocked hostnames
const BLOCKED_HOSTS = new Set([
  "localhost",
  "127.0.0.1",
  "::1",
  "0.0.0.0",
  "metadata.google.internal",
  "169.254.169.254"
]);

function isPrivateIP(hostname: string): boolean {
  if (BLOCKED_HOSTS.has(hostname.toLowerCase())) return true;
  
  // IPv4 private patterns
  const ipv4Match = hostname.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (ipv4Match) {
    const b0 = parseInt(ipv4Match[1], 10);
    const b1 = parseInt(ipv4Match[2], 10);
    
    // 127.0.0.0/8 (Loopback)
    if (b0 === 127) return true;
    // 10.0.0.0/8 (Private)
    if (b0 === 10) return true;
    // 172.16.0.0/12 (Private)
    if (b0 === 172 && b1 >= 16 && b1 <= 31) return true;
    // 192.168.0.0/16 (Private)
    if (b0 === 192 && b1 === 168) return true;
    // 169.254.0.0/16 (Link Local)
    if (b0 === 169 && b1 === 254) return true;
    // 0.0.0.0/8
    if (b0 === 0) return true;
  }

  // Block local domains
  if (hostname.endsWith(".local") || hostname.endsWith(".internal") || hostname.endsWith(".onion")) {
    return true;
  }

  return false;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { url } = body;

    if (!url || typeof url !== "string") {
      return NextResponse.json({ error: "Missing or invalid url parameter." }, { status: 400 });
    }

    let parsedUrl: URL;
    try {
      parsedUrl = new URL(url);
    } catch {
      return NextResponse.json({ error: "Invalid URL format." }, { status: 400 });
    }

    // Enforce HTTP / HTTPS protocol
    if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
      return NextResponse.json({ error: "Only HTTP and HTTPS protocols are permitted." }, { status: 400 });
    }

    // SSRF validation
    if (isPrivateIP(parsedUrl.hostname)) {
      return NextResponse.json(
        { error: "Security Exception: Access to private, loopback, or metadata addresses is strictly prohibited (SSRF Protection)." },
        { status: 403 }
      );
    }

    // Fetch with timeout and size limit (Max 5MB)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000); // 8-second timeout

    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        "User-Agent": "AIoT_HW2_CustomDatasetFetcher/1.0",
        "Accept": "text/csv, text/plain, */*"
      }
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      return NextResponse.json(
        { error: `Remote server responded with HTTP status ${response.status}: ${response.statusText}` },
        { status: 502 }
      );
    }

    // Check content length header if present
    const contentLength = response.headers.get("content-length");
    if (contentLength && parseInt(contentLength, 10) > 5 * 1024 * 1024) {
      return NextResponse.json(
        { error: "File size exceeds the 5MB maximum limit." },
        { status: 413 }
      );
    }

    const text = await response.text();
    if (text.length > 5 * 1024 * 1024) {
      return NextResponse.json(
        { error: "File content exceeds the 5MB maximum limit." },
        { status: 413 }
      );
    }

    return NextResponse.json({
      success: true,
      url,
      sizeBytes: text.length,
      csvText: text
    });
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : "Unknown fetch failure";
    return NextResponse.json(
      { error: `Failed to fetch external CSV: ${errorMsg}` },
      { status: 500 }
    );
  }
}
