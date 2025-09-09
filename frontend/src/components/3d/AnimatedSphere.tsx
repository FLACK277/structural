import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh } from 'three';

interface AnimatedSphereProps {
  position?: [number, number, number];
  color?: string;
  radius?: number;
}

const AnimatedSphere = ({ 
  position = [0, 0, 0], 
  color = "#8b5cf6", 
  radius = 0.5 
}: AnimatedSphereProps) => {
  const meshRef = useRef<Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = state.clock.elapsedTime * 0.4;
      meshRef.current.position.x = position[0] + Math.cos(state.clock.elapsedTime) * 0.3;
      meshRef.current.position.z = position[2] + Math.sin(state.clock.elapsedTime * 0.7) * 0.3;
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[radius, 32, 32]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
};

export default AnimatedSphere;