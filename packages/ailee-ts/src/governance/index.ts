// Copyright (c) Don Michael Feeney Jr.
// Licensed under the MIT License.

/**
 * AILEE Governance v1 Data Contracts and Types
 */

export type CompartmentType = 'safety' | 'grace' | 'consensus' | 'fallback' | 'global';

export interface LedgerEntry {
  id: string;
  compartment: CompartmentType;
  inputSnapshot?: Record<string, unknown>;
  input_snapshot?: Record<string, unknown>;
  decision: Record<string, unknown>;
  metadata: Record<string, unknown>;
  previousHash?: string;
  previous_hash?: string;
  currentHash?: string;
  current_hash?: string;
  createdAt?: string;
  created_at?: string;
}

export interface DecisionSummary {
  id: string;
  compartment: CompartmentType;
  payload: Record<string, unknown>;
}

export interface FinalApprovalRequest {
  safetyDecision: DecisionSummary;
  graceDecision: DecisionSummary;
  consensusDecision: DecisionSummary;
  fallbackDecision: DecisionSummary;
}

export interface FinalApprovalResponse {
  approved: boolean;
  reason: string;
  policyId: string;
}

export interface CorrelatedEvent {
  index: number;
  request_id?: string;
  safety_entry_id: string;
  grace_entry_id: string;
  consensus_entry_id: string;
  fallback_entry_id: string;
  alignment_issues: string[];
  features: Record<string, unknown>;
  training_signals_count: number;
}

export interface LedgerDiffResult {
  comp1: string;
  comp1_count: number;
  comp2: string;
  comp2_count: number;
  comp1_latest_hash: string | null;
  comp2_latest_hash: string | null;
  count_difference: number;
}

export interface LedgerMirrorConfig {
  baseUrl: string;
}

/**
 * Client for consuming AILEE Governance endpoints from Python REST services.
 */
export class LedgerMirrorClient {
  private baseUrl: string;

  constructor(config: LedgerMirrorConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
  }

  async getLedgerEntries(
    compartment: CompartmentType,
    params?: { minRisk?: number; maxRisk?: number; limit?: number }
  ): Promise<{ compartment: string; total: number; entries: LedgerEntry[] }> {
    const url = new URL(`${this.baseUrl}/api/v1/ledger/${compartment}`);
    if (params?.minRisk !== undefined) url.searchParams.set('min_risk', params.minRisk.toString());
    if (params?.maxRisk !== undefined) url.searchParams.set('max_risk', params.maxRisk.toString());
    if (params?.limit !== undefined) url.searchParams.set('limit', params.limit.toString());

    const resp = await fetch(url.toString());
    if (!resp.ok) {
      throw new Error(`Failed to fetch ledger entries: ${resp.statusText}`);
    }
    return resp.json();
  }

  async getLedgerDiff(comp1: CompartmentType, comp2: CompartmentType): Promise<LedgerDiffResult> {
    const url = new URL(`${this.baseUrl}/api/v1/ledger/diff`);
    url.searchParams.set('comp1', comp1);
    url.searchParams.set('comp2', comp2);

    const resp = await fetch(url.toString());
    if (!resp.ok) {
      throw new Error(`Failed to fetch ledger diff: ${resp.statusText}`);
    }
    return resp.json();
  }

  async getEventCorrelation(): Promise<{
    total_correlated_events: number;
    correlated_events: CorrelatedEvent[];
    global_ledger_count: number;
  }> {
    const url = `${this.baseUrl}/api/v1/events/correlation`;
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to fetch event correlations: ${resp.statusText}`);
    }
    return resp.json();
  }

  async submitFinalApproval(req: FinalApprovalRequest): Promise<FinalApprovalResponse> {
    const url = `${this.baseUrl}/api/v1/approval/evaluate`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!resp.ok) {
      throw new Error(`Failed to evaluate final approval: ${resp.statusText}`);
    }
    return resp.json();
  }
}
