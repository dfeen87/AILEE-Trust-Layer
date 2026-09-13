//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

export type CommandType = "SET_FLOW" | "SET_PRESSURE" | "PURGE_LINE" | "VALVE_CLOSE" | "VALVE_HOLD";

export interface ActuationCommand {
  targetDeviceId: string;
  commandType: CommandType;
  payloadValue: number; // Target flow rate (SLPM/SCCM) or target pressure (PSI)
  timestamp: number; // Unix timestamp ms
  requestingAgentId: string;
  gasId?: string | number; // Required for SET_FLOW if gas selection changes
}

export function validateActuationCommand(data: unknown): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  if (typeof data !== "object" || data === null) {
    return { valid: false, errors: ["Command payload must be a non-null object"] };
  }

  const c = data as Record<string, unknown>;

  if (typeof c.targetDeviceId !== "string" || c.targetDeviceId.trim() === "") {
    errors.push("targetDeviceId must be a non-empty string");
  }
  const validTypes: CommandType[] = ["SET_FLOW", "SET_PRESSURE", "PURGE_LINE", "VALVE_CLOSE", "VALVE_HOLD"];
  if (!validTypes.includes(c.commandType as CommandType)) {
    errors.push(`commandType must be one of: ${validTypes.join(", ")}`);
  }
  if (typeof c.payloadValue !== "number" || Number.isNaN(c.payloadValue)) {
    errors.push("payloadValue must be a valid number");
  }
  if (typeof c.timestamp !== "number" || Number.isNaN(c.timestamp) || c.timestamp <= 0) {
    errors.push("timestamp must be a valid positive epoch timestamp");
  }
  if (typeof c.requestingAgentId !== "string" || c.requestingAgentId.trim() === "") {
    errors.push("requestingAgentId must be a non-empty string");
  }

  return { valid: errors.length === 0, errors };
}
