//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import { DeviceStatus, MFCDeviceTelemetry, validateMFCTelemetry } from "../types/telemetry.js";
import { DEFAULT_MANIFEST, FieldbusManifest } from "./ethernet_ip.js";
import { canonicalizeGasId } from "../models/gas_database.js";

/**
 * EtherCAT Compact Cyclic PDO Parser & Serializer.
 * EtherCAT (CoE / CANopen over EtherCAT) uses Little-Endian native byte order for process data objects (PDOs).
 * Decoupled via hardware configuration manifest.
 */
export class EtherCATAdapter {
  public static parsePDOFrame(buffer: ArrayBuffer, manifest: FieldbusManifest = DEFAULT_MANIFEST): MFCDeviceTelemetry {
    if (buffer.byteLength < manifest.frameSizeBytes) {
      throw new Error(`EtherCAT buffer underflow: expected at least ${manifest.frameSizeBytes} bytes, got ${buffer.byteLength}`);
    }

    const view = new DataView(buffer);
    const offsets = manifest.etherCAT.byteOffsets;

    const flowRate = view.getFloat32(offsets.flowRate, true); // true = Little-Endian
    const setpoint = view.getFloat32(offsets.setpoint, true);
    const valvePosition = view.getFloat32(offsets.valvePosition, true);
    const temperature = view.getFloat32(offsets.temperature, true);
    const zeroOffset = view.getFloat32(offsets.zeroOffset, true);
    const gasId = view.getUint16(offsets.gasId, true);
    const statusFlags = view.getUint16(offsets.statusFlags, true);

    const predictiveScoreOffset = offsets.predictiveScore ?? 24;
    const predictiveScore = buffer.byteLength >= predictiveScoreOffset + 1 ? view.getUint8(predictiveScoreOffset) : 0;

    let deviceStatus: DeviceStatus = "OK";
    if ((statusFlags & 0x8000) !== 0) {
      deviceStatus = "FAULT";
    } else if ((statusFlags & 0x4000) !== 0) {
      deviceStatus = "WARN";
    }

    return {
      flowRate,
      setpoint,
      valvePosition,
      temperature,
      zeroOffset,
      gasId,
      deviceStatus,
      statusFlags,
      predictiveScore,
    };
  }

  public static serializePDOFrame(telemetry: MFCDeviceTelemetry, manifest: FieldbusManifest = DEFAULT_MANIFEST): ArrayBuffer {
    const validation = validateMFCTelemetry(telemetry);
    if (!validation.valid) {
      throw new Error(`Invalid EtherCAT telemetry: ${validation.errors.join("; ")}`);
    }
    const buffer = new ArrayBuffer(manifest.frameSizeBytes);
    const view = new DataView(buffer);
    const offsets = manifest.etherCAT.byteOffsets;

    view.setFloat32(offsets.flowRate, telemetry.flowRate, true);
    view.setFloat32(offsets.setpoint, telemetry.setpoint, true);
    view.setFloat32(offsets.valvePosition, telemetry.valvePosition, true);
    view.setFloat32(offsets.temperature, telemetry.temperature, true);
    view.setFloat32(offsets.zeroOffset, telemetry.zeroOffset, true);
    view.setUint16(offsets.gasId, canonicalizeGasId(telemetry.gasId), true);

    let flags = telemetry.statusFlags || 0;
    if (telemetry.deviceStatus === "FAULT") flags |= 0x8000;
    if (telemetry.deviceStatus === "WARN") flags |= 0x4000;
    view.setUint16(offsets.statusFlags, flags, true);

    const predictiveScoreOffset = offsets.predictiveScore ?? 24;
    if (buffer.byteLength > predictiveScoreOffset) {
      let scoreVal = 0;
      if (typeof telemetry.predictiveScore === "number" && Number.isFinite(telemetry.predictiveScore)) {
        scoreVal = telemetry.predictiveScore <= 1.0 ? Math.round(telemetry.predictiveScore * 255) : Math.round(telemetry.predictiveScore);
        scoreVal = Math.max(0, Math.min(255, scoreVal));
      }
      view.setUint8(predictiveScoreOffset, scoreVal);
    }

    return buffer;
  }
}
