import React, { useEffect, useRef } from 'react';
import { useGLTF } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';

const MODEL_PATH = '/models/avatar.glb?v=101';

const BONE_ALIASES = {
  spine: ['Spine', 'mixamorigSpine'],
  spine1: ['Spine1', 'mixamorigSpine1'],
  spine2: ['Spine2', 'mixamorigSpine2'],
  neck: ['Neck', 'mixamorigNeck'],
  head: ['Head', 'mixamorigHead'],
  rightShoulder: ['RightShoulder', 'mixamorigRightShoulder'],
  leftShoulder: ['LeftShoulder', 'mixamorigLeftShoulder'],
  rightArm: ['RightArm', 'mixamorigRightArm', 'arm_r'],
  leftArm: ['LeftArm', 'mixamorigLeftArm', 'arm_l'],
  rightForeArm: ['RightForeArm', 'mixamorigRightForeArm'],
  leftForeArm: ['LeftForeArm', 'mixamorigLeftForeArm'],
  rightHand: ['RightHand', 'mixamorigRightHand'],
  leftHand: ['LeftHand', 'mixamorigLeftHand'],
};

const FINGER_CHAINS = {
  right: [
    ['RightHandThumb1', 'mixamorigRightHandThumb1'],
    ['RightHandThumb2', 'mixamorigRightHandThumb2'],
    ['RightHandThumb3', 'mixamorigRightHandThumb3'],
    ['RightHandIndex1', 'mixamorigRightHandIndex1'],
    ['RightHandIndex2', 'mixamorigRightHandIndex2'],
    ['RightHandIndex3', 'mixamorigRightHandIndex3'],
    ['RightHandMiddle1', 'mixamorigRightHandMiddle1'],
    ['RightHandMiddle2', 'mixamorigRightHandMiddle2'],
    ['RightHandMiddle3', 'mixamorigRightHandMiddle3'],
    ['RightHandRing1', 'mixamorigRightHandRing1'],
    ['RightHandRing2', 'mixamorigRightHandRing2'],
    ['RightHandRing3', 'mixamorigRightHandRing3'],
    ['RightHandPinky1', 'mixamorigRightHandPinky1'],
    ['RightHandPinky2', 'mixamorigRightHandPinky2'],
    ['RightHandPinky3', 'mixamorigRightHandPinky3'],
  ],
  left: [
    ['LeftHandThumb1', 'mixamorigLeftHandThumb1'],
    ['LeftHandThumb2', 'mixamorigLeftHandThumb2'],
    ['LeftHandThumb3', 'mixamorigLeftHandThumb3'],
    ['LeftHandIndex1', 'mixamorigLeftHandIndex1'],
    ['LeftHandIndex2', 'mixamorigLeftHandIndex2'],
    ['LeftHandIndex3', 'mixamorigLeftHandIndex3'],
    ['LeftHandMiddle1', 'mixamorigLeftHandMiddle1'],
    ['LeftHandMiddle2', 'mixamorigLeftHandMiddle2'],
    ['LeftHandMiddle3', 'mixamorigLeftHandMiddle3'],
    ['LeftHandRing1', 'mixamorigLeftHandRing1'],
    ['LeftHandRing2', 'mixamorigLeftHandRing2'],
    ['LeftHandRing3', 'mixamorigLeftHandRing3'],
    ['LeftHandPinky1', 'mixamorigLeftHandPinky1'],
    ['LeftHandPinky2', 'mixamorigLeftHandPinky2'],
    ['LeftHandPinky3', 'mixamorigLeftHandPinky3'],
  ],
};

function getBone(nodes, aliases) {
  return aliases.map((name) => nodes[name]).find(Boolean) || null;
}

function cloneRotation(rotation) {
  return { x: rotation.x, y: rotation.y, z: rotation.z };
}

function ensureRestRotation(store, key, bone) {
  if (!bone) return null;
  if (!store[key]) {
    store[key] = cloneRotation(bone.rotation);
  }
  return store[key];
}

function applyRotationOffset(bone, baseRotation, offset, blend = 0.35) {
  if (!bone || !baseRotation) return;

  const targetX = baseRotation.x + (offset?.x ?? 0);
  const targetY = baseRotation.y + (offset?.y ?? 0);
  const targetZ = baseRotation.z + (offset?.z ?? 0);

  bone.rotation.x += (targetX - bone.rotation.x) * blend;
  bone.rotation.y += (targetY - bone.rotation.y) * blend;
  bone.rotation.z += (targetZ - bone.rotation.z) * blend;
}

function applyFingerCurl(nodes, side, curl, restRotations, blend = 0.35) {
  const direction = side === 'left' ? -1 : 1;

  FINGER_CHAINS[side].forEach((aliases, index) => {
    const bone = getBone(nodes, aliases);
    if (!bone) return;

    const restKey = aliases[0];
    const baseRotation = ensureRestRotation(restRotations, restKey, bone);
    const thumbBias = aliases[0].includes('Thumb') ? 0.45 : 1;
    const segmentBias = index % 3 === 0 ? 0.9 : 1.1;
    const curlOffset = direction * curl * thumbBias * segmentBias;

    applyRotationOffset(
      bone,
      baseRotation,
      { x: curlOffset, y: 0, z: curlOffset * 0.08 },
      blend
    );
  });
}

export default function Avatar({ animationData }) {
  const { scene, nodes } = useGLTF(MODEL_PATH);
  const frameRef = useRef(0);
  const timerRef = useRef(0);
  const restRotationsRef = useRef({});

  useEffect(() => {
    if (!nodes) return;

    console.log('3D model loaded successfully.');
    console.log('Detected Bones:', Object.keys(nodes));

    const capturedRotations = {};

    Object.entries(BONE_ALIASES).forEach(([key, aliases]) => {
      const bone = getBone(nodes, aliases);
      if (bone) {
        capturedRotations[key] = cloneRotation(bone.rotation);
      }
    });

    Object.values(FINGER_CHAINS).forEach((fingerAliases) => {
      fingerAliases.forEach((aliases) => {
        const bone = getBone(nodes, aliases);
        if (bone) {
          capturedRotations[aliases[0]] = cloneRotation(bone.rotation);
        }
      });
    });

    restRotationsRef.current = capturedRotations;
  }, [nodes]);

  useEffect(() => {
    frameRef.current = 0;
    timerRef.current = 0;
  }, [animationData]);

  useFrame((state, delta) => {
    if (!nodes) return;

    const restRotations = restRotationsRef.current;
    const hasAnimation = animationData && animationData.length > 0;

    timerRef.current += delta;
    if (timerRef.current < 1 / 30) return;
    timerRef.current = 0;

    const frameData = hasAnimation ? animationData[frameRef.current] : {};

    const spine = getBone(nodes, BONE_ALIASES.spine);
    const spine1 = getBone(nodes, BONE_ALIASES.spine1);
    const spine2 = getBone(nodes, BONE_ALIASES.spine2);
    const neck = getBone(nodes, BONE_ALIASES.neck);
    const head = getBone(nodes, BONE_ALIASES.head);
    const rightShoulder = getBone(nodes, BONE_ALIASES.rightShoulder);
    const leftShoulder = getBone(nodes, BONE_ALIASES.leftShoulder);
    const rightArm = getBone(nodes, BONE_ALIASES.rightArm);
    const leftArm = getBone(nodes, BONE_ALIASES.leftArm);
    const rightForeArm = getBone(nodes, BONE_ALIASES.rightForeArm);
    const leftForeArm = getBone(nodes, BONE_ALIASES.leftForeArm);
    const rightHand = getBone(nodes, BONE_ALIASES.rightHand);
    const leftHand = getBone(nodes, BONE_ALIASES.leftHand);

    applyRotationOffset(spine, ensureRestRotation(restRotations, 'spine', spine), frameData.spine_rotation);
    applyRotationOffset(spine1, ensureRestRotation(restRotations, 'spine1', spine1), frameData.spine_upper_rotation || frameData.spine_rotation);
    applyRotationOffset(spine2, ensureRestRotation(restRotations, 'spine2', spine2), frameData.spine_upper_rotation);
    applyRotationOffset(neck, ensureRestRotation(restRotations, 'neck', neck), frameData.neck_rotation, 0.3);
    applyRotationOffset(head, ensureRestRotation(restRotations, 'head', head), frameData.head_rotation, 0.28);
    applyRotationOffset(
      rightShoulder,
      ensureRestRotation(restRotations, 'rightShoulder', rightShoulder),
      frameData.right_shoulder_rotation
    );
    applyRotationOffset(
      leftShoulder,
      ensureRestRotation(restRotations, 'leftShoulder', leftShoulder),
      frameData.left_shoulder_rotation
    );
    applyRotationOffset(rightArm, ensureRestRotation(restRotations, 'rightArm', rightArm), frameData.right_arm_rotation);
    applyRotationOffset(leftArm, ensureRestRotation(restRotations, 'leftArm', leftArm), frameData.left_arm_rotation);
    applyRotationOffset(
      rightForeArm,
      ensureRestRotation(restRotations, 'rightForeArm', rightForeArm),
      frameData.right_forearm_rotation
    );
    applyRotationOffset(
      leftForeArm,
      ensureRestRotation(restRotations, 'leftForeArm', leftForeArm),
      frameData.left_forearm_rotation
    );
    applyRotationOffset(rightHand, ensureRestRotation(restRotations, 'rightHand', rightHand), frameData.right_hand_rotation);
    applyRotationOffset(leftHand, ensureRestRotation(restRotations, 'leftHand', leftHand), frameData.left_hand_rotation);

    applyFingerCurl(nodes, 'right', frameData.right_hand_curl ?? 0, restRotations, 0.32);
    applyFingerCurl(nodes, 'left', frameData.left_hand_curl ?? 0, restRotations, 0.32);

    if (hasAnimation) {
      frameRef.current = (frameRef.current + 1) % animationData.length;
    }
  });

  return <primitive object={scene} scale={2} position={[0, -2, 0]} />;
}

useGLTF.preload(MODEL_PATH);
