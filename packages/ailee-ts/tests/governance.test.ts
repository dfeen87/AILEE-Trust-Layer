import { describe, it, expect } from 'vitest';
import { LedgerMirrorClient, VERSION } from '../src/index';

describe('TypeScript Governance Mirror v9.4.0', () => {
  it('should export correct version', () => {
    expect(VERSION).toBe('9.4.0');
  });

  it('should instantiate LedgerMirrorClient with custom baseUrl', () => {
    const client = new LedgerMirrorClient({ baseUrl: 'http://localhost:8000' });
    expect(client).toBeDefined();
  });
});
