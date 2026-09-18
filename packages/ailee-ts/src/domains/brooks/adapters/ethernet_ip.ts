//! Copyright (c) Don Michael Feeney Jr.
//! Licensed under the MIT License.

import manifestJson from "../configs/sla5800_manifest.json" assert { type: "json" };
import { canonicalizeGasId } from "../models/gas_database.js";
import { DeviceStatus, MFCDeviceTelemetry, validateMFCTelemetry } from "../types/telemetry.js";

export interface FieldbusManifest {
  deviceName: string;
  frameSizeBytes: number;
  ethernetIP: {
    endianness: string;
    byteOffsets: {
      flowRate: number;
      setpoint: number;
      valvePosition: number;
      temperature: number;
      zeroOffset: number;
      gasId: number;
      statusFlags: number;
      predictiveScore?: number;
    };
  };
  etherCAT: {
    endianness: string;
    byteOffsets: {
      flowRate: number;
      setpoint: number;
      valvePosition: number;
      temperature: number;
      zeroOffset: number;
      gasId: number;
      statusFlags: number;
      predictiveScore?: number;
    };
  };
}

export const DEFAULT_MANIFEST: FieldbusManifest = manifestJson as FieldbusManifest;

/**
 * EtherNet/IP (CIP) Fieldbus Parser & Serializer.
 * EtherNet/IP CIP application data is encoded in little-endian byte order.
 * Decoupled via hardware configuration manifest.
 */
export class EtherNetIPAdapter {
  public static parseCIPFrame(buffer: ArrayBuffer, manifest: FieldbusManifest = DEFAULT_MANIFEST): MFCDeviceTelemetry {
    if (buffer.byteLength < manifest.frameSizeBytes) {
      throw new Error(`EtherNet/IP buffer underflow: expected at least ${manifest.frameSizeBytes} bytes, got ${buffer.byteLength}`);
    }

    const view = new DataView(buffer);
    const offsets = manifest.ethernetIP.byteOffsets;

    const flowRate = view.getFloat32(offsets.flowRate, true);
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

  public static serializeCIPFrame(telemetry: MFCDeviceTelemetry, manifest: FieldbusManifest = DEFAULT_MANIFEST): ArrayBuffer {
    const validation = validateMFCTelemetry(telemetry);
    if (!validation.valid) {
      throw new Error(`Invalid EtherNet/IP telemetry: ${validation.errors.join("; ")}`);
    }
    const buffer = new ArrayBuffer(manifest.frameSizeBytes);
    const view = new DataView(buffer);
    const offsets = manifest.ethernetIP.byteOffsets;

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
