import React, { forwardRef, useEffect, useRef } from 'react';
import { Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';

const JOINT_COLOR = '#00ffcc';
const BONE_COLOR = '#ffffff';
const JOINT_SIZE = 0.04;
const HEAD_RADIUS = 0.13;

const BoneNode = forwardRef(function BoneNode({ position, name, children }, ref) {
  return (
    <group ref={ref} position={position} name={name}>
      <mesh>
        <sphereGeometry args={[JOINT_SIZE, 16, 16]} />
        <meshBasicMaterial color={JOINT_COLOR} wireframe />
      </mesh>
      {children}
    </group>
  );
});

const HeadNode = forwardRef(function HeadNode({ position }, ref) {
  return (
    <group ref={ref} position={position} name="Head">
      <mesh>
        <ringGeometry args={[HEAD_RADIUS * 0.72, HEAD_RADIUS, 48]} />
        <meshBasicMaterial color={JOINT_COLOR} side={2} />
      </mesh>
      <mesh position={[0, 0.03, 0]}>
        <circleGeometry args={[JOINT_SIZE * 0.75, 24]} />
        <meshBasicMaterial color={JOINT_COLOR} />
      </mesh>
    </group>
  );
});

function BoneLine({ to }) {
  return (
    <group>
      <Line
        points={[
          [0, 0, 0],
          to,
        ]}
        color={BONE_COLOR}
        lineWidth={2.2}
        dashed
        dashSize={0.06}
        gapSize={0.025}
        transparent
        opacity={0.95}
      />
      <Line
        points={[
          [0, 0, 0],
          to,
        ]}
        color={JOINT_COLOR}
        lineWidth={0.7}
        transparent
        opacity={0.35}
      />
    </group>
  );
}

function applyRotation(ref, frameData, dataKey) {
  const rotation = frameData?.[dataKey];
  if (!ref.current || !rotation) return;

  ref.current.rotation.set(rotation.x ?? 0, rotation.y ?? 0, rotation.z ?? 0);
}

export default function Avatar({ animationData }) {
  const frameRef = useRef(0);
  const timerRef = useRef(0);
  const hasFinishedRef = useRef(false);

  const spineRef = useRef();
  const neckRef = useRef();
  const headRef = useRef();
  const rightShoulderRef = useRef();
  const leftShoulderRef = useRef();
  const rightArmRef = useRef();
  const leftArmRef = useRef();
  const rightForeArmRef = useRef();
  const leftForeArmRef = useRef();
  const rightHandRef = useRef();
  const leftHandRef = useRef();

  useEffect(() => {
    frameRef.current = 0;
    timerRef.current = 0;
    hasFinishedRef.current = false;
  }, [animationData]);

  useFrame((state, delta) => {
    const hasAnimation = Array.isArray(animationData) && animationData.length > 0;

    timerRef.current += delta;
    if (timerRef.current < 1 / 30) return;
    timerRef.current = 0;

    if (!hasAnimation) return;

    const frameData = animationData[frameRef.current];

    applyRotation(spineRef, frameData, 'spine_rotation');
    applyRotation(neckRef, frameData, 'neck_rotation');
    applyRotation(headRef, frameData, 'head_rotation');

    applyRotation(rightShoulderRef, frameData, 'right_shoulder_rotation');
    applyRotation(rightArmRef, frameData, 'right_arm_rotation');
    applyRotation(rightForeArmRef, frameData, 'right_forearm_rotation');
    applyRotation(rightHandRef, frameData, 'right_hand_rotation');

    applyRotation(leftShoulderRef, frameData, 'left_shoulder_rotation');
    applyRotation(leftArmRef, frameData, 'left_arm_rotation');
    applyRotation(leftForeArmRef, frameData, 'left_forearm_rotation');
    applyRotation(leftHandRef, frameData, 'left_hand_rotation');

    if (!hasFinishedRef.current && frameRef.current < animationData.length - 1) {
      frameRef.current += 1;
    } else {
      frameRef.current = animationData.length - 1;
      hasFinishedRef.current = true;
    }
  });

  return (
    <group position={[0, -2, 0]} scale={2}>
      <BoneNode ref={spineRef} name="Spine" position={[0, 1, 0]}>
        <BoneLine to={[0, 0.4, 0]} />
        <BoneNode ref={neckRef} name="Neck" position={[0, 0.4, 0]}>
          <BoneLine to={[0, 0.2, 0]} />
          <HeadNode ref={headRef} position={[0, 0.2, 0]} />
        </BoneNode>

        <BoneLine to={[-0.2, 0.3, 0]} />
        <BoneNode ref={rightShoulderRef} name="RightShoulder" position={[-0.2, 0.3, 0]}>
          <BoneLine to={[-0.2, 0, 0]} />
          <BoneNode ref={rightArmRef} name="RightArm" position={[-0.2, 0, 0]}>
            <BoneLine to={[0, -0.3, 0]} />
            <BoneNode ref={rightForeArmRef} name="RightForeArm" position={[0, -0.3, 0]}>
              <BoneLine to={[0, -0.3, 0]} />
              <BoneNode ref={rightHandRef} name="RightHand" position={[0, -0.3, 0]} />
            </BoneNode>
          </BoneNode>
        </BoneNode>

        <BoneLine to={[0.2, 0.3, 0]} />
        <BoneNode ref={leftShoulderRef} name="LeftShoulder" position={[0.2, 0.3, 0]}>
          <BoneLine to={[0.2, 0, 0]} />
          <BoneNode ref={leftArmRef} name="LeftArm" position={[0.2, 0, 0]}>
            <BoneLine to={[0, -0.3, 0]} />
            <BoneNode ref={leftForeArmRef} name="LeftForeArm" position={[0, -0.3, 0]}>
              <BoneLine to={[0, -0.3, 0]} />
              <BoneNode ref={leftHandRef} name="LeftHand" position={[0, -0.3, 0]} />
            </BoneNode>
          </BoneNode>
        </BoneNode>
      </BoneNode>
    </group>
  );
}
