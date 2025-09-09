import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { Suspense } from 'react';

interface Scene3DProps {
  children?: React.ReactNode;
  cameraPosition?: [number, number, number];
  className?: string;
  enableControls?: boolean;
}

const Scene3D = ({ 
  children, 
  cameraPosition = [0, 0, 5], 
  className = "w-full h-full",
  enableControls = false 
}: Scene3DProps) => {
  return (
    <div className={className}>
      <Canvas>
        <PerspectiveCamera position={cameraPosition} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <Suspense fallback={null}>
          {children}
        </Suspense>
        {enableControls && <OrbitControls />}
      </Canvas>
    </div>
  );
};

export default Scene3D;