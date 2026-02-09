import { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh, Color } from 'three';

interface InteractiveCubeProps {
  position?: [number, number, number];
  size?: number;
}

const InteractiveCube = ({ position = [0, 0, 0], size = 1 }: InteractiveCubeProps) => {
  const meshRef = useRef<Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const [clicked, setClicked] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = state.clock.elapsedTime * 0.5;
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.3;
      
      // Hover effect
      const scale = hovered ? 1.2 : 1;
      meshRef.current.scale.setScalar(scale);
      
      // Click effect
      if (clicked) {
        meshRef.current.rotation.z = state.clock.elapsedTime * 2;
      }
    }
  });

  return (
    <mesh
      ref={meshRef}
      position={position}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
      onClick={() => setClicked(!clicked)}
    >
      <boxGeometry args={[size, size, size]} />
      <meshStandardMaterial 
        color={hovered ? "#ff6b6b" : clicked ? "#4ecdc4" : "#6366f1"} 
        roughness={0.1}
        metalness={0.2}
      />
    </mesh>
  );
};

export default InteractiveCube;