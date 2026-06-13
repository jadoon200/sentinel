import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Default to 5173 (the documented `make ui` port) but honor a PORT env var so
  // preview/launch tooling can place the dev server on an assigned port.
  server: { port: process.env.PORT ? Number(process.env.PORT) : 5173 },
})
