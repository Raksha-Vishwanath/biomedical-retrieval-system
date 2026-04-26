import { NextRequest, NextResponse } from "next/server";

const BACKEND_API_BASE_URL =
  process.env.BACKEND_API_BASE_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.BIOSEEK_API_BASE_URL ??
  "http://127.0.0.1:8000/api";

function buildTargetUrl(path: string[], search: string) {
  const normalizedBase = BACKEND_API_BASE_URL.replace(/\/+$/, "");
  const normalizedPath = path.join("/");
  return `${normalizedBase}/${normalizedPath}${search}`;
}

async function forward(request: NextRequest, path: string[]) {
  const targetUrl = buildTargetUrl(path, request.nextUrl.search);
  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("connection");
  headers.delete("content-length");

  const init: RequestInit = {
    method: request.method,
    headers,
    redirect: "follow",
    cache: "no-store"
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = await request.text();
  }

  const response = await fetch(targetUrl, init);
  const payload = await response.arrayBuffer();

  return new NextResponse(payload, {
    status: response.status,
    headers: response.headers
  });
}

export async function GET(request: NextRequest, context: { params: { path: string[] } }) {
  return forward(request, context.params.path);
}

export async function POST(request: NextRequest, context: { params: { path: string[] } }) {
  return forward(request, context.params.path);
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 204 });
}
