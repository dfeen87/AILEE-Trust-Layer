//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export interface IdentityMembrane {
  sourceId: string;
  domain: string;
  signature?: string;
  verified: boolean;
}

export interface RoutingEnvelope<T = unknown> {
  id: string;
  identity: IdentityMembrane;
  timestamp: number;
  payload: T;
  routeTrace: string[];
}

export class RoutingMembraneEngine {
  private allowedDomains: Set<string>;

  constructor(allowedDomains: string[] = ["*"]) {
    this.allowedDomains = new Set(allowedDomains);
  }

  public verifyEnvelope<T>(envelope: RoutingEnvelope<T>): boolean {
    if (!envelope.identity || !envelope.identity.verified) {
      return false;
    }

    if (this.allowedDomains.has("*") || this.allowedDomains.has(envelope.identity.domain)) {
      envelope.routeTrace.push(`VERIFIED_AT_${Date.now()}`);
      return true;
    }

    envelope.routeTrace.push(`REJECTED_UNAUTHORIZED_DOMAIN_${envelope.identity.domain}`);
    return false;
  }
}
