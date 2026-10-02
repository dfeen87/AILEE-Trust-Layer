//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import gasDbJson from "../configs/gas_db.json" assert { type: "json" };

export type GasClassification = "INERT" | "FLAMMABLE" | "TOXIC" | "CORROSIVE" | "OXIDIZER" | "PYROPHORIC";

export interface GasDefinition {
  gasId: number | string;
  formula: string;
  name: string;
  gcf: number; // Gas Correction Factor / Thermal K-Factor relative to N2 (N2 = 1.0)
  classifications: GasClassification[];
  defaultMaxFlowSlpm: number;
  purgeRequiredOnChange: boolean;
}

export const GAS_DATABASE: Record<string | number, GasDefinition> = {};

// Populate GAS_DATABASE from JSON manifest
for (const entry of gasDbJson.gases as GasDefinition[]) {
  GAS_DATABASE[entry.gasId] = entry;
  GAS_DATABASE[entry.formula] = entry;
}

export function lookupGas(gasId: number | string): GasDefinition | undefined {
  return GAS_DATABASE[gasId];
}

export function canonicalizeGasId(gasId: number | string): number {
  const gas = lookupGas(gasId);
  if (!gas) {
    throw new Error(`Unknown or unregistered gas ID: ${gasId}`);
  }

  const canonical = typeof gas.gasId === "number" ? gas.gasId : Number(gas.gasId);
  if (!Number.isFinite(canonical)) {
    throw new Error(`Invalid canonical gas ID: ${gas.gasId}`);
  }

  return canonical;
}

export function isHazardousGas(gas: GasDefinition): boolean {
  return gas.classifications.some((c) => ["FLAMMABLE", "TOXIC", "CORROSIVE", "OXIDIZER", "PYROPHORIC"].includes(c));
}
