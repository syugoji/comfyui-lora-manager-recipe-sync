import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: [
      // web/comfyui/*.js はブラウザへ配信される絶対URL `/loras_static/...` で
      // static/ 配下を import する（サーバ側は lora_manager.py の
      // `add_static("/loras_static", config.static_path)` で解決している）。
      // vitest はこのルートを知らないため、同じ対応付けをここで与えないと
      // 収集時に "Failed to resolve import" で落ちる。
      {
        find: /^\/loras_static\//,
        replacement: fileURLToPath(new URL('./static/', import.meta.url))
      }
    ]
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['tests/frontend/setup.js'],
    include: [
      'tests/frontend/**/*.test.js',
      'tests/frontend/**/*.test.ts'
    ],
    coverage: {
      enabled: process.env.VITEST_COVERAGE === 'true',
      provider: 'v8',
      reporter: ['text', 'lcov', 'json-summary'],
      reportsDirectory: 'coverage/frontend'
    }
  }
});
