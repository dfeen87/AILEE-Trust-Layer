//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export type GasClassification = "INERT" | "FLAMMABLE" | "TOXIC" | "CORROSIVE" | "OXIDIZER" | "PYROPHORIC";

export interface GasDefinition {
  gasId: number | string;
  formula: string;
  name: string;
  gcf: number; // Gas Correction Factor relative to N2 (N2 = 1.0)
  classifications: GasClassification[];
  defaultMaxFlowSlpm: number;
  purgeRequiredOnChange: boolean;
}

export const GAS_DATABASE: Record<string | number, GasDefinition> = {
  1: {
    gasId: 1,
    formula: "N2",
    name: "Nitrogen",
    gcf: 1.0,
    classifications: ["INERT"],
    defaultMaxFlowSlpm: 1000,
    purgeRequiredOnChange: false,
  },
  "N2": {
    gasId: 1,
    formula: "N2",
    name: "Nitrogen",
    gcf: 1.0,
    classifications: ["INERT"],
    defaultMaxFlowSlpm: 1000,
    purgeRequiredOnChange: false,
  },
  2: {
    gasId: 2,
    formula: "Air",
    name: "Air",
    gcf: 1.0,
    classifications: ["INERT"],
    defaultMaxFlowSlpm: 1000,
    purgeRequiredOnChange: false,
  },
  "Air": {
    gasId: 2,
    formula: "Air",
    name: "Air",
    gcf: 1.0,
    classifications: ["INERT"],
    defaultMaxFlowSlpm: 1000,
    purgeRequiredOnChange: false,
  },
  3: {
    gasId: 3,
    formula: "Ar",
    name: "Argon",
    gcf: 1.415,
    classifications: ["INERT"],
    defaultMaxFlowSlpm: 1000,
    purgeRequiredOnChange: false,
  },
  "Ar": {
    gasId: 3,
    formula: "Ar",
    name: "Argon",
    gcf: 1.415,
    classifications: ["INERT"],
    defaultMaxFlowSlpm: 1000,
    purgeRequiredOnChange: false,
  },
  10: {
    gasId: 10,
    formula: "H2",
    name: "Hydrogen",
    gcf: 1.01,
    classifications: ["FLAMMABLE"],
    defaultMaxFlowSlpm: 100,
    purgeRequiredOnChange: true,
  },
  "H2": {
    gasId: 10,
    formula: "H2",
    name: "Hydrogen",
    gcf: 1.01,
    classifications: ["FLAMMABLE"],
    defaultMaxFlowSlpm: 100,
    purgeRequiredOnChange: true,
  },
  11: {
    gasId: 11,
    formula: "NH3",
    name: "Ammonia",
    gcf: 0.73,
    classifications: ["TOXIC", "CORROSIVE"],
    defaultMaxFlowSlpm: 50,
    purgeRequiredOnChange: true,
  },
  "NH3": {
    gasId: 11,
    formula: "NH3",
    name: "Ammonia",
    gcf: 0.73,
    classifications: ["TOXIC", "CORROSIVE"],
    defaultMaxFlowSlpm: 50,
    purgeRequiredOnChange: true,
  },
  15: {
    gasId: 15,
    formula: "O2",
    name: "Oxygen",
    gcf: 0.993,
    classifications: ["OXIDIZER"],
    defaultMaxFlowSlpm: 200,
    purgeRequiredOnChange: true,
  },
  "O2": {
    gasId: 15,
    formula: "O2",
    name: "Oxygen",
    gcf: 0.993,
    classifications: ["OXIDIZER"],
    defaultMaxFlowSlpm: 200,
    purgeRequiredOnChange: true,
  },
  28: {
    gasId: 28,
    formula: "SiH4",
    name: "Silane",
    gcf: 0.598,
    classifications: ["PYROPHORIC"],
    defaultMaxFlowSlpm: 20,
    purgeRequiredOnChange: true,
  },
  "SiH4": {
    gasId: 28,
    formula: "SiH4",
    name: "Silane",
    gcf: 0.598,
    classifications: ["PYROPHORIC"],
    defaultMaxFlowSlpm: 20,
    purgeRequiredOnChange: true,
  },
  42: {
    gasId: 42,
    formula: "Cl2",
    name: "Chlorine",
    gcf: 0.86,
    classifications: ["CORROSIVE", "TOXIC"],
    defaultMaxFlowSlpm: 30,
    purgeRequiredOnChange: true,
  },
  "Cl2": {
    gasId: 42,
    formula: "Cl2",
    name: "Chlorine",
    gcf: 0.86,
    classifications: ["CORROSIVE", "TOXIC"],
    defaultMaxFlowSlpm: 30,
    purgeRequiredOnChange: true,
  },
};

export function lookupGas(gasId: number | string): GasDefinition | undefined {
  return GAS_DATABASE[gasId];
}

export function isHazardousGas(gas: GasDefinition): boolean {
  return gas.classifications.some((c) => ["FLAMMABLE", "TOXIC", "CORROSIVE", "OXIDIZER", "PYROPHORIC"].includes(c));
}
