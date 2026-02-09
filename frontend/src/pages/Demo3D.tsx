import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import { Suspense } from 'react';
import InteractiveCube from '@/components/3d/InteractiveCube';
import WaveParticles from '@/components/3d/WaveParticles';
import GeometryMorph from '@/components/3d/GeometryMorph';
import FloatingCube from '@/components/3d/FloatingCube';
import AnimatedSphere from '@/components/3d/AnimatedSphere';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

const Demo3D = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="absolute top-0 left-0 w-full z-20 p-6">
        <div className="flex justify-between items-center">
          <Link to="/">
            <Button variant="outline" className="bg-black/20 backdrop-blur-md border-white/20 text-white hover:bg-white/10">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Home
            </Button>
          </Link>
          <h1 className="text-2xl font-bold text-white">3D Interactive Demo</h1>
        </div>
      </div>

      {/* 3D Scene */}
      <Canvas camera={{ position: [0, 0, 10], fov: 50 }}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#8b5cf6" />
        
        <Suspense fallback={
          <Html center>
            <div className="text-white text-xl">Loading 3D Scene...</div>
          </Html>
        }>
          {/* Interactive Elements */}
          <InteractiveCube position={[-3, 2, 0]} size={1.5} />
          <InteractiveCube position={[3, -2, 0]} size={1.2} />
          <InteractiveCube position={[0, 0, -3]} size={1} />
          
          {/* Morphing Geometry */}
          <GeometryMorph position={[-4, -1, 2]} color="#ec4899" />
          <GeometryMorph position={[4, 1, -2]} color="#06b6d4" />
          
          {/* Floating Objects */}
          <FloatingCube position={[2, 3, 2]} color="#f59e0b" size={0.8} />
          <AnimatedSphere position={[-2, -3, 2]} color="#10b981" radius={0.6} />
          
          {/* Particle Effects */}
          <WaveParticles />
          
          {/* Controls */}
          <OrbitControls 
            enablePan={true} 
            enableZoom={true} 
            enableRotate={true}
            autoRotate={true}
            autoRotateSpeed={0.5}
          />
        </Suspense>
      </Canvas>

      {/* Instructions */}
      <div className="absolute bottom-0 left-0 w-full z-20 p-6">
        <div className="bg-black/20 backdrop-blur-md rounded-lg p-4 text-white max-w-md">
          <h3 className="font-semibold mb-2">Interactive Controls:</h3>
          <ul className="text-sm space-y-1">
            <li>• Click and drag to rotate the scene</li>
            <li>• Scroll to zoom in/out</li>
            <li>• Click on cubes to interact with them</li>
            <li>• Hover over objects for effects</li>
          </ul>
        </div>
      </div>

      {/* Info Panel */}
      <div className="absolute top-20 right-6 z-20 bg-black/20 backdrop-blur-md rounded-lg p-4 text-white max-w-xs">
        <h3 className="font-semibold mb-2">3D Features:</h3>
        <ul className="text-sm space-y-1">
          <li>✓ Three.js with React</li>
          <li>✓ WebGL Rendering</li>
          <li>✓ Interactive Objects</li>
          <li>✓ Particle Systems</li>
          <li>✓ Real-time Animations</li>
          <li>✓ Dynamic Lighting</li>
        </ul>
      </div>
    </div>
  );
};

export default Demo3D;