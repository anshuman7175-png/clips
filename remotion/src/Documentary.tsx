import React from "react";
import {
  AbsoluteFill,
  Audio,
  Img,
  OffthreadVideo,
  Sequence,
  interpolate,
  useCurrentFrame,
} from "remotion";

/** Mirrors pipeline/edit/remotion_export.py::export_props */
export type SequenceProps = {
  eventId: string;
  src: string | null;
  missing: boolean;
  isStill: boolean;
  fromFrame: number;
  durationFrames: number;
  kind: "document" | "still" | "footage";
  cutStyle: "hard" | "j_cut" | "l_cut";
  hold: boolean;
  ambienceSwell: boolean;
  label: string | null;
  attribution: string | null;
};

export type DocumentaryProps = {
  fps: number;
  width: number;
  height: number;
  durationFrames: number;
  narrationSrc: string | null;
  sequences: SequenceProps[];
};

export const defaultProps: DocumentaryProps = {
  fps: 30,
  width: 1920,
  height: 1080,
  durationFrames: 150,
  narrationSrc: null,
  sequences: [],
};

/** Slow Ken Burns on stills/documents; held shots move even slower. */
const KenBurnsStill: React.FC<{ seq: SequenceProps }> = ({ seq }) => {
  const frame = useCurrentFrame();
  const range = seq.hold ? 0.04 : 0.08;
  const scale = interpolate(frame, [0, seq.durationFrames], [1.0, 1.0 + range], {
    extrapolateRight: "clamp",
  });
  return (
    <AbsoluteFill style={{ backgroundColor: "#000", overflow: "hidden" }}>
      <Img
        src={seq.src as string}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: `scale(${scale})`,
        }}
      />
    </AbsoluteFill>
  );
};

const MissingSlate: React.FC<{ seq: SequenceProps }> = ({ seq }) => (
  <AbsoluteFill
    style={{
      backgroundColor: "#101010",
      alignItems: "center",
      justifyContent: "center",
      color: "#e5e5e5",
      fontFamily: "monospace",
      fontSize: 48,
    }}
  >
    {`MISSING ${seq.eventId}`}
  </AbsoluteFill>
);

const Overlays: React.FC<{ seq: SequenceProps }> = ({ seq }) => (
  <>
    {seq.label ? (
      <div
        style={{
          position: "absolute",
          left: 32,
          bottom: 32,
          padding: "8px 16px",
          backgroundColor: "rgba(0,0,0,0.55)",
          color: "rgba(255,255,255,0.92)",
          fontFamily: "Arial, sans-serif",
          fontSize: 28,
          letterSpacing: 4,
        }}
      >
        {seq.label}
      </div>
    ) : null}
    {seq.attribution ? (
      <div
        style={{
          position: "absolute",
          right: 32,
          bottom: 32,
          color: "rgba(255,255,255,0.6)",
          fontFamily: "Arial, sans-serif",
          fontSize: 20,
        }}
      >
        {seq.attribution}
      </div>
    ) : null}
  </>
);

export const Documentary: React.FC<DocumentaryProps> = (props) => (
  <AbsoluteFill style={{ backgroundColor: "#000" }}>
    {props.sequences.map((seq) => (
      <Sequence
        key={seq.eventId}
        from={seq.fromFrame}
        durationInFrames={seq.durationFrames}
      >
        {seq.missing || !seq.src ? (
          <MissingSlate seq={seq} />
        ) : seq.isStill ? (
          <KenBurnsStill seq={seq} />
        ) : (
          <AbsoluteFill style={{ backgroundColor: "#000" }}>
            <OffthreadVideo
              src={seq.src}
              muted
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
          </AbsoluteFill>
        )}
        <Overlays seq={seq} />
      </Sequence>
    ))}
    {props.narrationSrc ? <Audio src={props.narrationSrc} /> : null}
  </AbsoluteFill>
);
