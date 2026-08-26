import { Config } from "@remotion/cli/config";

Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);
// FFmpeg encode happens downstream (PLAN.md Layer 8); keep intermediate high quality.
Config.setCodec("h264");
Config.setCrf(16);
