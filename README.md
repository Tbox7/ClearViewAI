## Inspiration  
Navigating the world can be difficult for individuals with visual impairments. Our inspiration came from the idea of giving sight through intelligence — using AI to turn cameras into real-time assistants. With Azure’s powerful AI tools and GitHub Copilot, we built an interface to help users “see” their surroundings in a meaningful way.
## What It Does  
Clearview AI is a real-time, multi-modal AI assistant designed to run on smart glasses. It:
- Reads text from the environment (like signs or labels)  
- Identifies the object directly in front of the user  
- Recognizes known faces and announces their name aloud  
- Speaks everything out loud using Azure Text-to-Speech  
Users simply press a key (`r`, `i`, `f`, `q`) to control what the assistant does.
 ## How We Built It  
- Azure Computer Vision API for OCR and object detection  
- Azure Speech API for Text-to-Speech feedback  
- face_recognition and OpenCV for real-time facial recognition  
- GitHub Copilot to accelerate development, generate class scaffolds, and help debug issues  
- Modular Python classes to allow scaling or adapting to wearables like Meta Ray-Bans
## Challenges We Ran Into  
- Installing and aligning all Azure SDKs and face_recognition dependencies was tricky across environments  
- Getting accurate object positioning required tuning to ensure only the most centered object was considered  
- Handling edge cases where no faces were detected, or multiple were partially visible, required stability logic
## Accomplishments That We're Proud Of  
- Built a fully functional smart assistant that runs in real-time from just a webcam feed  
- Integrated three major AI modalities: text, image, and voice  
- Achieved solid modular structure that’s easy to deploy, extend, or reuse for other AI wearable projects
## What We Learned  
- How to harness Azure AI’s capabilities effectively, including batching API calls and managing async operations  
- Real-time face encoding and recognition under webcam constraints  
- How to structure AI pipelines cleanly across computer vision, voice, and user interaction
## What’s Next for Clearview AI  
- Add voice-command support for hands-free operation  
- Use on-device ML to allow offline fallback  
- Integrate GPS and object labeling to assist with navigation  
- Deploy directly on Meta Smart Glasses
