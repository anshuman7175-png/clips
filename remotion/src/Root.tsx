import React from "react";
import { Composition } from "remotion";
import { Documentary, type DocumentaryProps, defaultProps } from "./Documentary";

export const Root: React.FC = () => (
  <Composition
    id="Documentary"
    component={Documentary}
    width={1920}
    height={1080}
    fps={30}
    durationInFrames={defaultProps.durationFrames}
    defaultProps={defaultProps}
    calculateMetadata={({ props }) => {
      const p = props as DocumentaryProps;
      return {
        durationInFrames: p.durationFrames,
        fps: p.fps,
        width: p.width,
        height: p.height,
      };
    }}
  />
);
