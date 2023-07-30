import * as THREE from 'three';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth /
window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );


const geometry = new THREE.SphereGeometry(2, 32, 32);


const textureLoader = new THREE.TextureLoader();
const texture = textureLoader.load('/earthmap1k.jpg');

const material = new THREE.MeshBasicMaterial({ map: texture });


const earth = new THREE.Mesh(geometry, material);




camera.position.z = 6;


function animate() {


  requestAnimationFrame( animate );
    earth.rotation.x += 0.01;
    earth.rotation.y += 0.01;
    updateTextPosition();
  renderer.render( scene, camera );
}
animate();