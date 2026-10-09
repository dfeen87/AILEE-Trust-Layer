import { describe, it, expect } from 'vitest';
import { LedgerMirrorClient, VERSION } from '../src/index';

describe('TypeScript Governance Mirror v10.0.2', () => {
  it('should export correct version', () => {
    expect(VERSION).toBe('10.0.2');
  });

  it('should instantiate LedgerMirrorClient with custom baseUrl', () => {
    const client = new LedgerMirrorClient({ baseUrl: 'http://localhost:8000' });
    expect(client).toBeDefined();
  });
});
