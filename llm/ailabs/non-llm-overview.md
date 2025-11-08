# 📚 Non-LLM Foundation Models: Vision, Audio, Video, Robotics & Scientific

This document tracks foundation models and specialized builders across non-LLM domains: image generation, video synthesis, audio/speech generation, robotics, scientific models, and medical AI.

---

## 🎨 Image & Vision Generation Models

### European Image Gen Labs

- [x] 🇩🇪 Black Forest Labs
  - 🎨 FLUX.1 series (12B parameters)
  - 🎨 FLUX.1 Schnell (open-source, fastest)
  - 🎨 FLUX.1 Dev (open-weight)
  - 🎨 FLUX.1 Pro (premium API)
  - 🎨 FLUX 1.1 Pro (ultra mode)
  - 💰 $4B valuation talks (2024)
  - 👥 Founded by ex-Stability AI team

### Chinese Image Gen Labs

- [x] 🇨🇳 Tencent - Hunyuan Image 3.0
  - 🎨 Hunyuan Image 3.0: Text-to-image generation
  - 🎨 Supports multiple styles (watercolor, oil painting, animation, 3D)
  - 📊 Chinese and English support
  - 🎭 Different Dimension Me: Anime photo generator
  - 🔓 Parts of Hunyuan open-sourced (3D models, translation tools)
  - 📱 Available on Hunyuan-image.com

### Multimodal Companies with Image Gen (See LLM-Overview for full details)
- Google (Gemini Diffusion, Veo video)
- OpenAI (DALL-E via GPT-4V)
- Stability AI (Stable Diffusion series)
- Meta (Imagine/Emu image models)

---

## 🎬 Video Generation Models

### Frontier Video Gen (US Dominance)

- [x] 🇺🇸 OpenAI - Sora/Sora 2
  - 🎥 Sora: 10-20 second videos, 480p-1080p, text/image-to-video
  - 🎥 Sora 2 (Sep 2025): Up to 1 minute, 16-bit HDR, synchronized audio, enhanced physics
  - 💰 Available: ChatGPT Plus ($20/mo, 50 videos), Pro ($200/mo, higher res)
  - 🏆 Diffusion transformer architecture

- [x] 🇺🇸 Runway - Gen-2, Gen-3, Gen-4
  - 🎥 Gen-2 (2023): 4-second videos, text/image-to-video
  - 🎥 Gen-3 Alpha (2024): Up to 10 seconds, expressive humans, 60-90s generation
  - 🎥 Gen-4 (March 2025): 5-10 seconds, world consistency, physics simulation
  - 🎥 Gen-4 Turbo (April 2025): 5x faster generation
  - 💰 $536.5M funding, Series D: $308M at $3B valuation (2025)
  - 📊 12T+ tokens training

- [x] 🇺🇸 Google DeepMind - Veo/Veo 2/Veo 3
  - 🎥 Veo (May 2024): 1080p, 60+ seconds, cinematography-aware
  - 🎥 Veo 2 (Dec 2024): Up to 4K, minutes-long videos
  - 🎥 Veo 3 (May 2025): 8-second videos with synchronized audio, lip-syncing
  - 🎥 Veo 3.1 (2025): Available via Gemini API, VideoFX
  - 📊 State-of-the-art physics simulation

- [x] 🇺🇸 Meta - Movie Gen/Emu Video
  - 🎥 Emu Video (Nov 2023): 4-second videos, 512x512, text/image-to-video
  - 🎥 Movie Gen (Oct 2024): 16 seconds at 1080p, synchronized audio (up to 45 seconds)
  - 📊 Video model: 30B params, Audio model: 13B params
  - 📊 Training: 100M video-text pairs, 1B image-text pairs
  - 🏆 Outperforms Runway Gen-3 and Sora in benchmarks (research only)

- [x] 🇺🇸 Midjourney - V1 Video
  - 🎥 V1 Video (June 2025): Image-to-video via "Animate"
  - 🎥 5-second base clips, extendable by 4s up to 4 times (max 21 seconds)
  - 🎨 Multiple styles: live-action, stop-motion, animation, VFX
  - 💰 $10/month ($10 cheaper than competitors, per Midjourney claims)
  - ⚡ 8x more compute than standard image gen

- [x] 🇺🇸 Pika - Pika 2.0/2.2
  - 🎥 Pika 2.0 (Dec 2024): 8-second videos, improved text alignment, motion rendering
  - 🎥 Pika 2.2 (2025): 10-second 1080p videos, Pikaframes keyframing
  - 💰 $141M funding, Series B: $80M at $470-700M valuation
  - 🎯 Scene Ingredients: upload/customize characters, objects, settings

- [x] 🇺🇸 Luma AI - Dream Machine/Ray
  - 🎥 Dream Machine 1.6: 5 seconds at 24fps, text/image-to-video, camera control
  - 🎥 Ray3 (2025): State-of-the-art physics, 16-bit HDR, 5x faster/cheaper
  - 💰 $173M total, Series C: $90M (Dec 2024) led by Amazon
  - 🏆 First 16-bit HDR video generation

### Specialized Avatar/Synthetic Video (Enterprise Focus)

- [x] 🇬🇧 Synthesia
  - 🎬 Deep learning for text-to-video + facial animation
  - 🗣️ 140+ languages/accents, AI avatars with natural expressions
  - 💰 $536M total, Series D: $180M at $2.1-4B valuation (Jan 2025)
  - 👥 60%+ of Fortune 100 use Synthesia
  - 🎯 Enterprise-focused, production-ready

- [x] 🇺🇸 HeyGen
  - 🎬 Avatar 3.0/Avatar IV with diffusion-inspired audio-to-expression engine
  - 🗣️ 175+ languages/dialects, 100+ AI voices
  - 📊 Photorealistic facial movements, voice-sync with hand gestures
  - 🏢 Relocated from China to Los Angeles (2022)

- [x] 🇮🇱 D-ID
  - 🎬 Creative Reality Studio, 3D facial modeling, RAG for conversations
  - 🗣️ 119 languages/dialects, up to 5-minute videos
  - 💰 $48M total funding
  - 👥 200M+ videos generated, 280K+ developers

### Asian Video Gen

- [x] 🇨🇳 Kuaishou - Kling AI
  - 🎥 Kling 1.0/1.6/2.0/2.1: Up to 2-minute videos at 30fps, 1080p
  - 🎨 DiT (Diffusion Transformer) + 3D VAE architecture
  - 📊 Text/image-to-video, various aspect ratios
  - 🏆 Positioned as "world's most powerful" video generator (Chinese competitor to OpenAI)

- [x] 🇨🇳 Shengshu Technology - Vidu
  - 🎥 Vidu Q1/Q2: 16-second to 2-minute videos, 1080p resolution
  - 🎬 Vidu Q2 (upgraded): 3x faster reference video generation
  - 🎨 Reference image generation: Supports inputting up to 7 images simultaneously
  - 🎯 Image-to-video capabilities with high consistency
  - 💰 Series A: Billions of yuan funding (2024)
  - 🤝 Investors: Baidu, Alibaba Ant Group, Qiming Venture Partners, Beijing AI Industry Investment Fund
  - 📊 Commercial success: 400M+ videos generated, $20M ARR
  - 🤝 Partnerships: JD.com, Amazon, e-commerce/advertising/animation verticals
  - 👨‍🔬 Founded March 2023 (spinout from Tsinghua University)

---

## 🎙️ Audio & Speech Foundation Models

### Voice & Speech Generation (Proprietary)

- [x] 🇺🇸 🇬🇧 ElevenLabs
  - 🗣️ Eleven v3 (latest): Most expressive with emotional control
  - 🗣️ Eleven Turbo v2.5, Flash v2.5 (75ms ultra-low latency)
  - 📊 70+ languages (v3) vs 29 in v2
  - 💬 Text-to-Dialogue API for multi-speaker conversations
  - 💰 $3.3B valuation (Series C Jan 2025)
  - 🏆 Transformer-based, context-aware speech synthesis

- [x] 🇺🇸 Microsoft VALL-E / Azure AI Speech
  - 🗣️ VALL-E 2: Achieves human parity zero-shot TTS
  - 🗣️ VALL-E X: Multilingual, cross-lingual synthesis
  - 🗣️ DragonV2.1 Neural: Personal voice models
  - 📊 100+ languages, voice cloning from 3 seconds
  - 🎯 Neural codec language model architecture

- [x] 🇺🇸 Meta Voicebox / Audiobox
  - 🗣️ Voicebox: Flow Matching model, voice editing, style transfer
  - 🗣️ Audiobox: Multi-modal generation (voice + text prompts), 160k+ hours training
  - 📊 50k+ hours multilingual (6 languages)
  - 🏆 20x faster than VALL-E, 10x more intelligible
  - ⚠️ Research only, not publicly released

- [x] 🇺🇸 PlayAI / Meta Acquisition
  - 🗣️ PlayHT 2.0: 1M+ hours training, 10x larger model
  - 🗣️ Play 3.0-mini: 30+ languages, optimized for AI agents
  - 📊 Sub-800ms latency, real-time voice cloning (3-10 seconds)
  - 💰 Meta acquired PlayAI July 2025
  - 🌍 100+ languages with emotional expressiveness

- [x] 🇺🇸 Cartesia - Sonic/Sonic 2/Sonic 3
  - 🗣️ Sonic 3: State Space Models (SSM) architecture, 3x faster than transformers
  - ⚡ Ultra-low latency: 40ms (Turbo), 90ms (standard)
  - 📊 42+ languages, emotional expression (laughter, emotion)
  - 🎯 First SSM alternative to Transformers for TTS
  - 💰 Founded 2023 by Stanford AI Lab alumni

- [x] 🇺🇸 Descript - Overdub (Lyrebird acquisition)
  - 🗣️ Overdub Voice: Custom voice cloning
  - 🗣️ Rapid Voice Clone 2.0 (2024): 20 seconds needed
  - 📊 Voice Design: text-to-voice generation
  - 💰 $101M total funding, Lyrebird acquired Sept 2019

- [x] 🇨🇦 Resemble AI
  - 🗣️ Rapid Voice Clone 2.0 (2024): 20 seconds audio needed
  - 🗣️ Deepfake detection (Resemble Detect)
  - 💰 $12M funding, Series A: $8M (2023)

- [x] 🇺🇸 Speechify (WWDC 2025 Apple Design Award)
  - 🗣️ 1,000+ natural voices, 60+ languages
  - 👥 50M+ users
  - 🎯 AI Voice Cloning, Dubbing, Voice Changer

- [x] 🇨🇳 iFlyTek (科大讯飞) - Unisound
  - 🗣️ Speech recognition, synthesis, voiceprint synthesis
  - 🗣️ Offline control and far-field processing
  - 📊 Anti-noise and far-field speech signal processing
  - 🎯 Use cases: Robotics, education, healthcare, connected vehicles
  - 🏢 Major Chinese AI company, publicly listed
  - 🤝 Partnerships with Huawei, Xiaomi, other major tech firms

### Music & Sound Generation

- [x] 🇺🇸 Suno
  - 🎵 Chirp v1-v5: Text-to-music with 1,200+ genres
  - 🎵 v5 (Sept 2025): Up to 8 minutes of music, 90% prompt adherence
  - 🎵 Suno Studio (2025): Audio workstation with stem editing
  - 📊 Multimodal transformer + latent diffusion
  - 💰 $125M Series B at $500M valuation
  - ⚖️ RIAA copyright infringement lawsuit

- [x] 🇺🇸 Udio
  - 🎵 Allegro v1.5 (Oct 2025): Cleaner vocals, cohesive harmonies, precise control
  - 🎵 Multi-lingual support, extended track creation
  - 💰 $10M seed, led by Andreessen Horowitz
  - ⚖️ RIAA copyright infringement lawsuit

- [x] 🇬🇧 Stability AI - Stable Audio
  - 🎵 Stable Audio Open: 1.1B params (full), 341M (compact)
  - 🎵 Stable Audio 2.0/2.5: Up to 47 seconds, full tracks with coherent structures
  - 🎵 2.5 (Sept 2025): Enterprise-grade, 8-step generation (50 steps previously)
  - 📊 Latent diffusion with DiT, compressed autoencoder
  - 💰 $225M total, eliminated debt 2024

- [x] 🇺🇸 Meta - MusicGen / AudioCraft
  - 🎵 MusicGen: 300M, 1.5B, 3.3B parameter variants
  - 🎵 AudioGen: Text-to-sound effects
  - 📊 EnCodec neural codec + transformer autoregressive LM
  - 📊 Training: 20k hours (400k recordings)
  - 🔓 Open-source

- [x] 🇺🇸 Google - MusicLM / Magenta
  - 🎵 MusicLM (2023): Hierarchical sequence-to-sequence
  - 🎵 Magenta RealTime (June 2025): 800M params, 48kHz stereo streaming
  - 📊 24kHz music for several minutes (MusicLM), real-time streaming (RT)
  - 🎯 Research focus

- [x] 🇨🇳 ByteDance - Seed-Music / MeLoDy / StemGen
  - 🎵 Seed-Music (2024): Multimodal (text, audio, scores, sound prompts)
  - 🎵 MeLoDy: LM-guided diffusion, 257k hours training
  - 🎵 StemGen: End-to-end generation, 500 hours licensed music
  - 📱 Ripple app integration (iOS, US beta)

- [x] 🇺🇸 Adobe - Project Music / Firefly Audio
  - 🎵 Project Music GenAI Control (research preview, Feb 2024)
  - 🎵 Firefly Audio (Oct 2025): Commercial, studio-quality tracks
  - 🎵 Full licensing, adjust tempo/structure/intensity/length
  - 🔧 8-step generation breakthrough (2025)

- [x] 🇺🇸 Riffusion
  - 🎵 Fine-tuned Stable Diffusion on spectrograms
  - 🎵 Real-time generation, latent space interpolation
  - 💰 $4M seed (Oct 2023), Advisors: The Chainsmokers
  - 🔓 Open-source approach

- [x] 🇦🇺 Splash Music
  - 🎵 HummingLM: Hum-to-full-track generation
  - 📊 Trained on AWS Trainium + SageMaker HyperPod
  - 💰 $20.1M funding
  - 🏆 50% faster training, 54% cost reduction

- [x] 🇱🇺 AIVA (Artificial Intelligence Virtual Artist)
  - 🎵 250+ musical styles, 85% accuracy in stylistic nuances
  - 🎵 Trained on 4 centuries of classical masterpieces
  - 🎯 Orchestral arrangement optimization

- [x] 🇺🇸 Boomy
  - 🎵 10M+ songs created (as of 2022)
  - 📱 Distribution to Spotify, Apple Music, TikTok, etc.
  - 🤝 ADA Worldwide (Warner Music Group) partnership

- [x] 🇯🇵 Soundraw
  - 🎵 Click-based prompting (genre, mood, tempo, instruments)
  - 💰 $3M Series A (March 2024)

- [x] 🌐 Mubert
  - 🎵 150+ channel categories, text-to-music, image-to-music
  - 🎵 Real-time unique generation (not database)

---

## 🤖 Robotics & Embodied AI Foundation Models

### US Humanoid Leaders

- [x] 🇺🇸 Tesla - Optimus
  - 🤖 **Foundation Model:** End-to-end neural network + Neural World Simulator
  - 🎯 Leverages Full Self-Driving (FSD) AI stack (48 neural networks)
  - 💾 **Training Data:** 1.5+ petabytes from vehicle fleet
  - 🎯 **Vision:** Vision-based only (no LiDAR)
  - 🧠 **Capabilities:** Reinforcement learning, human-like movement, real-time decisions
  - 📊 **Production:** 5,000 units targeted for 2025
  - ⚙️ **Transfer Learning:** From autonomous vehicles to robotics

- [x] 🇺🇸 Figure AI - Figure-02
  - 🤖 **Foundation Model:** Vision-Language Model (VLM) + Vision-Language-Action
  - 🤖 **Partner:** OpenAI (custom AI models partnership)
  - 📊 **Vision:** 6 RGB cameras + onboard VLM
  - 🏃 **Physical Specs:** 168cm, 70kg, 5h battery, 1.2m/s, 16 DOF hands
  - 🎯 **Capabilities:** 3x more compute than Figure-01, autonomous conversation
  - 💰 **Funding:** $675M Series B, $39.5B valuation talks (Feb 2025)
  - 💼 **Deployment:** BMW factory pilot

- [x] 🇺🇸 Physical Intelligence - π0 (pi-zero)
  - 🤖 **Architecture:** 3.3B total (3B PaliGemma + 300M control)
  - 🔧 **Innovation:** Novel "flow matching" architecture, 50Hz control
  - 📊 **Training:** 7 different robots, 68 tasks, 1-20h fine-tuning per new task
  - 🏆 **Capabilities:** First autonomous laundry folder, box assembly, bussing dishes
  - 💰 **Funding:** $400M (Nov 2024)
  - 🔓 **Status:** Open-sourced (GitHub: openpi)

- [x] 🇺🇸 Agility Robotics - Digit
  - 🤖 **Foundation Model:** <1M param LSTM "motor cortex"
  - 📊 **Training:** 2,000 hours simulated motion (3-4 days real-time in NVIDIA Isaac Sim)
  - 🎯 **Zero-shot sim-to-real:** >99% success on real hardware
  - 💼 **Deployments:** GXO Georgia fulfillment, Schaeffler plant (Cheraw, SC)
  - 🤝 **Partner:** NVIDIA Isaac Sim training

- [x] 🇺🇸 Apptronik - Apollo
  - 🤖 **Foundation Model:** Integrates NVIDIA Project GR00T VLM
  - 📊 **Physical Specs:** 173cm, 73kg, 25kg payload
  - 🎯 **Learning:** From text, video, demonstrations
  - 💼 **Partnership:** Mercedes-Benz testing

- [x] 🇺🇸 NVIDIA - GR00T N1.5
  - 🤖 **Architecture:** Dual-system (fast reflexive + slow deliberate reasoning)
  - 📊 **Specs:** 3B parameters, first open humanoid FM (2025)
  - ⚙️ **Hardware:** Runs on Jetson on-device
  - 📊 **Training:** 2 trillion tokens, heterogeneous data (real, video, synthetic)
  - 🤝 **Partnerships:** Figure AI, Apptronik, Agility, Boston Dynamics, Fourier
  - 🔓 **Status:** Open-source (GitHub: NVIDIA/Isaac-GR00T)

- [x] 🇬🇧 🇺🇸 1X Technologies - NEO
  - 🤖 **Platform:** EVE (commercial), NEO (consumer home robot)
  - 🤖 **Foundation Model:** Redwood AI (vision-language transformer)
  - 🌍 **World Model:** 1X World Model (data-driven simulator, physics-grounded)
  - 🎯 **Capabilities:** Voice commands, natural language, laundry, door answering
  - 💰 **Founder:** Trond Riiber Knudsen (ex-Halodi)
  - 💰 **Funding:** $23.5M Series A2 (March 2023, OpenAI Startup Fund)
  - 🏆 **World's first consumer humanoid with home deployment**

- [x] 🇺🇸 Sanctuary AI - Phoenix
  - 🤖 **Foundation Model:** Carbon AI Control System (mimics brain subsystems)
  - 📊 **Dexterity:** 20 DOF hands with proprietary haptic tech
  - ⚡ **Task Automation:** <24 hours (vs weeks previously)
  - 🎯 **Capabilities:** 100+ tasks across 12+ industries, natural language tasking
  - 📊 **Physical:** 170cm, 70kg, 25kg payload

- [x] 🇨🇦 Skild AI (CMU Spinout)
  - 🤖 **Foundation Model:** Skild Brain (scalable across hardware/tasks)
  - 👨‍🔬 **Founders:** CMU Profs Deepak Pathak & Abhinav Gupta
  - 📊 **Training:** 1,000x more data than competitors (claimed)
  - 💰 **Funding:** $300M (2024)
  - 🤖 **Platforms:** Mobile manipulation, quadruped

- [x] 🇮🇱 Mentee Robotics
  - 🤖 **Foundation Model:** Transformer-based LLM for task interpretation
  - 👨‍🔬 **Founder:** Prof. Amnon Shashua (AI expert)
  - 🎯 **Approach:** "Learning from data" + "Learning from experience"
  - 🎯 **Features:** Verbal command execution, 360-degree vision, marker-less navigation

### Chinese Humanoid (Rapid Scale-Up)

- [x] 🇨🇳 UBTech - Walker S1/S2
  - 🤖 **Foundation Model:** DeepSeek-R1 deep reasoning multimodal model
  - 🧠 **BrainNet:** "super brain" + "intelligent sub-brain" architecture
  - 📊 **Training:** High-quality industrial dataset from real factories
  - 🏆 **World-First:** Multi-humanoid robot coordination in factories (swarm intelligence)
  - 💼 **Deployments:** BYD, Audi China, Zeekr 5G smart factory
  - 📦 **Orders:** 500+ for Walker S1
  - 📈 **2025 Target:** 500-1K units (60%+ Walker S2)

- [x] 🇨🇳 AgiBot (Zhiyuan Robotics)
  - 🤖 **Foundation Model:** Genie Operator-1 (GO-1) - generalist embodied FM
  - 👨‍🔯 **Founders:** Deng Taihua & Peng Zhihui (ex-Huawei engineers)
  - 📊 **Dataset:** AgiBot World (1M+ training sets, 100 robots, open-source)
  - 🏆 **Largest humanoid manipulation dataset** (as of Dec 2024)
  - 📦 **Production:** 962 units manufactured (Dec 15, 2024)
  - 🏭 **Status:** Mass production begun
  - 🎯 **Goal:** Match Tesla Optimus output in 2025

- [x] 🇨🇳 Fourier Intelligence - GR-1
  - 🤖 **Foundation Model:** NVIDIA GR00T N1 support
  - 📊 **Physical Specs:** 165cm, 55kg, 40 DOF, 50kg payload
  - 🏆 **Claim:** World's first mass-produced humanoid (100+ units)
  - 📅 **Launch:** July 2023 World AI Conference Shanghai

- [x] 🇨🇳 Huawei - Kuafu Robot (Pangu 5.0 Embodied AI)
  - 🤖 **Foundation Model:** Pangu 5.0 (billions-trillions parameters)
  - 📊 **Architecture:** Understanding, NLP, task planning, dual-arm, autonomous execution
  - ☁️ **CloudRobo:** Deploys algorithms to cloud for lightweight on-robot processing
  - 🏢 **Innovation Center:** Shenzhen Global Embodied AI (2024)
  - 💰 **Investment:** $413M into robotics subsidiary
  - 🎯 **Goal:** "Humanoid Robot+" open ecosystem
  - 🤝 **Partnerships:** Leju Robot, Han's Robot

- [x] 🇨🇳 Unitree Robotics - G1 Humanoid
  - 🤖 **G1 Specs:** $16K (lowest-cost humanoid), superhuman flexibility
  - 🎯 **Capabilities:** Martial arts maneuvers, aerial cartwheels, kip-ups
  - 📚 **Tech Base:** Years of quadruped accumulation → bipedal in 6 months (2023)
  - 🎯 **CEO Prediction:** General-purpose robotic AI model by end 2025
  - 📈 **Launch:** August 2024

- [x] 🇨🇳 Ex-Robots (Dalian) - Entertainment Humanoids
  - 🤖 **Foundation Model:** Multi-modal for environment recognition + facial feedback
  - 👨‍🔬 **Founder:** Li Boyang (PhD AI, Waseda Univ 2010)
  - 🎯 **Focus:** Lifelike facial expressions
  - 📦 **Production:** 200+ operational, target 500+ by end 2024
  - 💰 **Price:** $210K-$280K per unit
  - 🤝 **Partnerships:** Huawei, iFlyTek, China Mobile

- [x] 🇨🇳 Xiaomi - CyberOne
  - 🤖 **Foundation Model:** Mi-Sense depth vision + AI interaction algorithm
  - 📊 **Physical Specs:** 177cm, 52kg, 21 DOF
  - 🎯 **Capabilities:** 3D perception, gesture/emotion recognition, 45 emotion classifications
  - 🔄 **Display:** Curved OLED for interactive information
  - 🎯 **Future:** Manufacturing integration for specific scenarios

### Quadruped Robots

- [x] 🇺🇸 Boston Dynamics - Spot/Atlas
  - 🤖 **Foundation Model:** Large Behavior Models (LBMs) partnership with TRI
  - 📊 **LBM Specs:** 450M-param diffusion transformer, 30Hz action output
  - 🎯 **Capabilities:** Rigid + deformable manipulation, T-shirt folding, assembly
  - 🤝 **Partnership:** Toyota Research Institute (Oct 2024)
  - 🏆 **Industry leader in dynamic mobility**

- [x] 🇨🇭 ANYbotics - ANYmal
  - 🤖 **Foundation Model:** Attention-based recurrent encoder + neural network policy
  - 📊 **Approach:** Sim-to-real transfer, fast automated data generation
  - 🎯 **Capabilities:** Jumping, climbing, crouching, parkour navigation
  - 🌐 **Community:** Hundreds of contributors (universities + corporate)
  - 🏆 **Strong sim-to-real, robust outdoor operation**

### Research & Academic Robotics

- [x] 🇺🇸 Google DeepMind - RT-1/RT-2/RT-2-X
  - 🤖 **RT-2:** First Vision-Language-Action (VLA) model (55B params)
  - 📊 **RT-2-X:** 3x more successful than RT-2 for emergent skills
  - 📊 **Training:** Web-scale vision-language + robotic data
  - 🎯 **Capabilities:** Novel object generalization, emergent reasoning
  - 🏆 **Pioneered VLA paradigm in robotics**

- [x] 🇺🇸 Stanford - Mobile ALOHA/OpenVLA
  - 🤖 **Mobile ALOHA:** Bimanual mobile manipulation ($32K, vs $200K commercial)
  - 📊 **Training:** Supervised behavior cloning, 50 demos/task, 90% success with co-training
  - 🎯 **Tasks:** Shrimp sauté, elevator operation, pan rinsing
  - 🤖 **OpenVLA:** 7B params, beats RT-2-X (55B) by 16.5% with 7x fewer params
  - 📊 **Training:** Open X-Embodiment dataset, 29 tasks, multiple embodiments

- [x] 🇺🇸 Berkeley RAIL - BridgeData V2/CrossFormer
  - 📊 **BridgeData V2:** Large-scale robot manipulation dataset
  - 📊 **CrossFormer:** 900K trajectories across 20 embodiments
  - 🔬 **Hardware:** WidowX 250 6DOF arm
  - 🏆 **Major cross-embodiment learning contributor**

- [x] 🇺🇸 MIT CSAIL - Foundation Model Supervision
  - 🤖 **KALM:** Pre-trained VLMs for task-relevant keypoints
  - 🤖 **Neural Jacobian Fields (NJF):** Robots learn body response via vision only
  - 🎯 **Approaches:** Leverage non-robot FMs for scalable supervision
  - 💪 **Impact:** 5-10 demonstrations sufficient for policy generalization

- [x] 🇺🇸 Toyota Research Institute - Large Behavior Models (LBMs)
  - 🤖 **Architecture:** Diffusion-based transformer (450M params)
  - 📊 **Training:** ~1,700 hours real robot + 47K sim rollouts, 1,800 evaluations
  - 🏆 **2024 Breakthrough:** 80% less data, learns 3-5x faster
  - 🤖 **Single LBM:** Hundreds of manipulation tasks
  - 🤖 **Output:** 50 DOF control at 30Hz
  - 💡 **Motto:** "If you can demonstrate it, the robot can learn it"
  - 🤝 **Partnership:** Boston Dynamics Atlas (Oct 2024)

- [x] 🇨🇭 ETH Zurich Robotics
  - 🔬 **Labs:** ASL, Robotics & Perception, IRIS (7 labs)
  - 🎯 **Focus:** Autonomous navigation (camera/computation only), control strategies
  - 📚 **Research:** Flying, service, mobile robots

- [x] 🇨🇳 Tencent - Robotics X Lab / Tairos Platform
  - 🤖 **Foundation Model:** Tairos (modular, plug-and-play embodied AI)
  - 📊 **SLAP3 Framework:** Sensing, Learning, Action, Planning, Perception models
  - ☁️ **Infrastructure:** Cloud simulation, training, data management
  - 🎯 **Vision:** Neutral platform for startups
  - 🤝 **Partnerships:** AgiBot, KEENON, Unitree
  - 💰 **Investor:** Agibot

### Foundation Model Frameworks (Infrastructure)

- [x] **Covariant - RFM-1 (Robotics Foundation Model)**
  - 🤖 **Architecture:** 8B parameter any-to-any transformer
  - 📊 **Training:** 4 years warehouse pick-and-place data + internet data
  - 🎯 **Capabilities:** First commercial GenAI for robots, physics world model
  - 📹 **AI-Generated Video:** Predicts future scenarios, selects best action
  - 💼 **Commercial Deployment:** Warehouse operations
  - 🏆 **First to give robots deeper understanding via GenAI**

- [x] **Open X-Embodiment Dataset**
  - 📊 **Scale:** 1M+ trajectories, 22 robots, 21 institutions worldwide
  - 🎯 **Impact:** Foundation for cross-embodiment learning

### Robotics Key Trends (2025)

- **Production Scale-Up:** Tesla 5K, UBTech 500-1K, AgiBot mass production
- **Cost Dropping:** Unitree G1 at $16K, Unitree CEO predicts <$10K by 2026
- **Chinese Acceleration:** UBTech multi-robot coordination, AgiBot 1M dataset, Huawei Pangu
- **VLA Model Convergence:** 2-7B parameters becoming standard (vs 55B for RT-2-X)
- **Model Efficiency:** Smaller models (7B OpenVLA) beating larger (55B RT-2-X) by 16.5%
- **Multi-Robot Coordination:** UBTech achieved world-first factory swarm intelligence
- **Data Efficiency:** Toyota LBMs: 80% less data, 3-5x faster learning
- **Sim-to-Real:** Zero-shot transfer >99% success (Agility Digit)
- **Massive Funding:** $1B+ rounds becoming common
- **Task Learning Speed:** <24 hours automation setup (Sanctuary Phoenix)

---

## 🧬 Scientific & Biological Foundation Models

### Protein & Molecular Biology

- [x] 🇬🇧 🇺🇸 DeepMind/Google - AlphaFold Series
  - 🧬 AlphaFold 3 (May 2024): Pairformer + diffusion model
  - 📊 Protein-complex prediction with DNA, RNA, post-translational mods, ligands
  - 📊 214M+ structures in public database
  - 💾 Proprietary training data (not disclosed)
  - 🏆 50%+ improvement over existing methods for interactions

- [x] 🇺🇸 OpenProtein Foundation - PoET/PoET-2
  - 🧬 PoET-2: 182M parameters (trillion-param performance equivalent)
  - 📊 Zero-shot indel prediction, clinical variant effects
  - 📊 30-fold less experimental data vs existing
  - 🏆 State-of-the-art for zero-shot predictions

- [x] 🇺🇸 EvolutionaryScale - ESM3/Evo/Evo 2
  - 🧬 ESM3: 98B parameters, 25x more FLOPs than ESM2
  - 🧬 Evo: 7B params, 131k context (prokaryotic genomes)
  - 🧬 Evo 2 (2025): 7B/40B params, 1M base pair context, 9.3T nucleotides from 128k+ genomes
  - 📊 StripedHyena architecture (3x faster than transformers)
  - 🏆 BRCA1 variant classification: 90% accuracy

- [x] 🇺🇸 Profluent Bio - ProGen3
  - 🧬 112M to 46B parameters (sparse protein LMs)
  - 📊 3.4B+ sequences, 1.5T tokens training
  - 🎯 OpenCRISPR-1 (first AI-designed genome editor)
  - 🎯 OpenAntibodies (rivaling blockbuster therapeutics)
  - 💰 $35M+ funding

- [x] 🇨🇭 🇺🇸 Genentech/Roche - Lab-in-the-Loop
  - 🧬 Proprietary foundation models for drug discovery
  - 💼 Therapeutic molecule design across all modalities
  - 🤝 NVIDIA partnership, Recursion $12B commitment

- [x] 🇺🇸 Schrödinger, Inc.
  - 🧬 Physics-based + ML (Free Energy Perturbation + ML)
  - 🎯 Molecular behavior prediction at atomic level
  - 🤝 NVIDIA DGX A100 systems for acceleration

- [x] 🇬🇧 Exscientia (acquired by Recursion Nov 2024)
  - 🧬 Design-Make-Test-Learn loops on AWS
  - 🎯 4.5 years → 12-15 months drug design
  - 💰 70% faster, 80% cost reduction

- [x] 🇺🇸 Recursion Pharmaceuticals - MolE/Phenom/Boltz-2
  - 🧬 MolE: Molecular foundation model (DeBERTa architecture)
  - 🧬 Phenom-Beta: Vision transformer for cellular microscopy
  - 🧬 Boltz-2 (with MIT): First model with structure + binding affinity
  - 💰 $50M NVIDIA investment, BioHive-2 supercomputer (504 H100s)

- [x] 🇺🇸 Insitro - Biomolecular Models
  - 🧬 ML models for ADMET, biomarker prediction
  - 🤝 Eli Lilly partnership (ADMET models), Mayo Clinic (ocular biomarkers)

- [x] 🇺🇸 Tempus AI - Multimodal Oncology FM
  - 🧬 $200M partnership with AstraZeneca + Pathos AI
  - 💾 8M+ patient records (1.4M imaging, 1.3M genomic, 260k transcriptomics)
  - 🎯 Largest multimodal oncology foundation model

- [x] 🇺🇸 Chai Discovery - Chai-1/Chai-2
  - 🧬 Multi-modal FM (proteins, DNA, RNA, small molecules)
  - 📊 Chai-1: 77% PoseBlast (vs AlphaFold3: 76%)
  - 📊 Chai-2: Atomic-level structure + binding prediction
  - 💰 $70M Series A (Aug 2024)

- [x] 🇺🇸 MIT + Recursion - Boltz-2
  - 🧬 First model combining structure AND binding affinity
  - 🏆 CASP16 ranked #1 on binding affinity prediction
  - 🔓 Open-source (MIT license for academic + commercial)

- [x] 🇺🇸 University of Washington - RoseTTAFold/RFdiffusion
  - 🧬 RoseTTAFold All-Atom (RFAA): Residue + atomic levels
  - 🧬 RFdiffusion All-Atom: Design proteins with binding pockets
  - 🔓 Open-source, free for all research + drug development
  - 🏆 Nobel Prize 2024 (David Baker)

### Earth Science & Climate Models

- [x] 🇬🇧 Google DeepMind - GraphCast/GenCast
  - 🌍 GraphCast: 10-day weather forecasts, 0.25° resolution
  - 🌍 GenCast (Dec 2024): Probabilistic 15-day forecasts, 99.8% accuracy >36hr
  - ⚡ <1 minute on single TPU v4 (vs hours on supercomputer)
  - 🏆 Hurricane Lee: 9-day landfall prediction (vs 6 days traditional)

- [x] 🇺🇸 Google Research - NeuralGCM
  - 🌍 Traditional fluid dynamics + neural networks for small-scale physics
  - 🌍 2-15 day forecasts, 40-year climate simulation
  - ⚡ 100,000x more efficient than X-SHiELD
  - 📱 Runs on single laptop

- [x] 🇺🇸 NVIDIA - Earth-2 / cBottle / CorrDiff
  - 🌍 cBottle: First generative AI climate foundation model at km resolution
  - 🌍 CorrDiff: Generative AI weather at km-scale
  - ⚡ 500x faster, 10,000x more energy-efficient

- [x] 🇺🇸 NASA + IBM - Prithvi Weather-Climate
  - 🌍 Prithvi WxC: 320M params (encoder 220M, decoder 100M)
  - 🌍 2.3B params version available
  - 📊 40 years NASA MERRA-2 data, 160 variables
  - 🔓 Open-source on Hugging Face

- [x] 🇨🇳 Huawei - Pangu-Weather
  - 🌍 10-day typhoon prediction, 5-day regional (3km resolution)
  - ⚡ 10 seconds on single GPU (vs 4-5 hours on 3k-server cluster)
  - 🏆 Successfully predicted Typhoon Saola (2023)
  - 🌾 Madagascar fishermen: 10-day vs 3-day traditional forecasts

- [x] 🇪🇺 ESA - TerraMind
  - 🌍 Multimodal earth observation (radar + optical + topography)
  - 💾 9M+ samples, 62TB raw data → 1TB optimized
  - 🤝 CloudFerro + ESA Φ-lab partnership

- [x] 🇺🇸 Microsoft Research - Aurora
  - 🌍 1.3B parameters, 3D Swin Transformer
  - 📊 Training: 1M+ hours weather/climate simulations
  - ⚡ 5,000x speedup vs traditional IFS
  - 🎯 First AI to predict global air pollution at km-scale

### Physics & Materials Science

- [x] 🇬🇧 Google DeepMind - GNoME
  - 🧪 2.2M new crystal structures discovered (~800 years equivalent knowledge)
  - 🏆 380k most stable candidates, 736 independently verified
  - 🎯 Superconductors, batteries (528 lithium-ion conductors), electronics

- [x] 🇺🇸 Meta FAIR - Universal Model for Atoms (UMA)
  - 🧪 Multi-size models for power/cost/speed tradeoffs
  - 📊 Open Molecules 2025: 100M quantum mechanics calculations
  - 🎯 Small molecules, biomolecules, metal complexes, electrolytes

- [x] 🇺🇸 Microsoft Research - MatterGen + MatterSim
  - 🧪 MatterGen: Novel material generation from requirements
  - 🧪 MatterSim: Energy, forces, stress prediction at finite T/P
  - 📊 MatterGen: 2.9x more stable structures, 17.5x closer to energy minimum
  - 🔓 Open-source (MIT license)

---

## 🏥 Medical & Healthcare Foundation Models

### Medical Imaging & Diagnostics

- [x] 🇺🇸 🇬🇧 Google DeepMind - Med-Gemini
  - 🏥 Med-Gemini: 91.1% on MedQA (USMLE-style), 4.6% improvement over Med-PaLM 2
  - 🏥 State-of-the-art on NEJM Image Challenges
  - 📊 Multimodal medical understanding (text + images)
  - 👁️ Superior to GPT-4V on medical benchmarks

- [x] 🇺🇸 Microsoft - Healthcare AI Models
  - 🏥 MedImageInsight: Embedding model for medical image analysis
  - 🏥 MedImageParse: Precise segmentation (X-ray, CT, MRI, ultrasound, pathology)
  - 🏥 CXRReportGen: Multimodal chest X-ray report generation
  - 🤝 Partnerships: Mass General Brigham, Mayo Clinic, University of Washington

- [x] 🇺🇸 Paige AI - PRISM2/Digital Pathology
  - 🏥 PRISM2: Foundation model connecting pathology images + clinical language
  - 🏥 Paige Prostate Detect: FDA de novo (Sept 2021)
  - 🏥 Paige PanCancer Detect: FDA Breakthrough (2024)
  - 🔬 Trained on large-scale pathology datasets

- [x] 🇺🇸 Providence Healthcare - Prov-GigaPath
  - 🏥 Pretrained on 1.3B pathology image tiles, 171k whole-slides
  - 📊 Largest whole-slide pretraining (5-10x TCGA)
  - 🏆 State-of-the-art on 25/26 digital pathology tasks
  - 📍 Available on Microsoft Azure AI Model Catalog

- [x] 🇩🇪 Aignostics - RudolfV / Atlas
  - 🏥 RudolfV: 103K slides, 750M patches, 60+ tissue types
  - 🏥 Atlas (with Mayo Clinic, Charité): 1.2M+ WSIs
  - 💰 €31.4M Series B (Oct 2024)
  - 🤝 Bayer strategic collaboration

- [x] 🇺🇸 PathAI - PLUTO
  - 🏥 PLUTO: Foundation model trained on 160k WSIs, 30+ disease areas
  - 📊 Multi-scale vision transformer, self-supervised learning
  - 🎯 Cellular, subcellular, and tissue-level analysis
  - 🤝 Roche expanded partnership (Feb 2024)

- [x] 🇮🇱 Aidoc - CARE1 Clinical AI Reasoning Engine
  - 🏥 CARE1: First clinical-grade CT foundation model
  - 🎯 Trained on millions of exams, adapts with minimal training
  - 📊 50+ FDA-cleared algorithms
  - 🏆 FDA Breakthrough for acute conditions in CT

- [x] 🇺🇸 Tempus AI - Multimodal Oncology (see Scientific section)
  - 🏥 $200M partnership with AstraZeneca + Pathos
  - 💾 8M+ de-identified patient records, multimodal data

- [x] 🇺🇸 🇬🇧 IBM Watson Health
  - 🏥 Watson Oncology: Cancer treatment recommendations
  - 🏥 Watsonx Foundation Models: Healthcare-specific fine-tuning

- [x] 🇬🇧 🇭🇲 🇩🇪 Siemens Healthineers
  - 🏥 VISTA-3D: CT segmentation (120+ organ classes)
  - 🏥 MAISI: Synthetic 3D CT image generation
  - 🤝 MONAI Deploy + NVIDIA BioNeMo integration

### Clinical Decision Support & EHR

- [x] 🇺🇸 OpenAI - GPT-4 Medical Applications
  - 🏥 Paradigm partnership: 10% accuracy over human experts on trial matching
  - 📊 Multimodal medical image interpretation
  - 💼 HIPAA-compliant APIs

- [x] 🇺🇸 Anthropic - Claude for Life Sciences
  - 🏥 Claude Sonnet 4.5: Superior medical imaging accuracy
  - 🏥 Sanofi partnership: Integrated into Concierge app
  - 🎯 Drug development support (discovery → commercialization)

- [x] 🇫🇷 Mistral AI - BioMistral
  - 🏥 BioMistral: Mistral 7B pre-trained on PubMed Central
  - 🌍 First large-scale multilingual medical LLM evaluation (7 languages)
  - ⚠️ Research tool only (NOT for production)

- [x] 🇺🇸 Meta - Me-LLaMA / Meditron
  - 🏥 Me-LLaMA: 13B/70B LLaMA 2-based, 129B medical tokens
  - 🏥 Meditron 7B/70B: Trained on clinical guidelines + papers
  - 📊 6 text analysis tasks + clinical diagnosis evaluation
  - 🔓 Open-source

- [x] 🇦🇪 M42 - Med42
  - 🏥 Med42: 70B parameters, 94.5% on USMLE sample exam (zero-shot)
  - 🏆 Surpasses prior open medical LLMs
  - 🔓 Free for non-commercial use (LLaMA 2-style license)

### Medical Imaging Foundation Models (Stanford, Harvard, etc.)

- [x] 🇺🇸 Stanford - CheXagent / RAD-DINO
  - 🏥 CheXagent: Chest X-ray interpretation (8 tasks), outperforms by up to 97.5%
  - 🏥 RAD-DINO: Biomedical image encoder (unimodal training)
  - 📖 CheXbench benchmark, CheXinstruct dataset

- [x] 🇺🇸 Harvard Medical School - Foundation Models
  - 🏥 Cancer imaging foundation model (Nature ML Intelligence, 2024)
  - 🏥 CONCH: Vision-language FM for pathology (1.17M image-text pairs)
  - 🏥 Chest X-ray, ECG, lung/heart sound models
  - 🤝 Broad Institute, Mass General Hospital collaboration

### Wearables & Health Monitoring

- [x] 🇺🇸 Apple Health - Biosignal Foundation Models
  - ⌚ PPG + ECG encoders with self-supervised learning
  - ⌚ Wearable Behavior Model (WBM): 92% accuracy predicting health conditions
  - 💾 Apple Heart Movement Study: 141K participants, 3 years
  - 🔒 On-device, end-to-end encrypted

- [x] 🇺🇸 Google Fitbit - Personal Health LLM
  - ⌚ Personal Health LLM: Based on Gemini, fine-tuned on de-identified signals
  - 🤖 Fitbit Labs Chatbot: Conversational Fitbit data queries
  - 💼 Device Connect: Enterprise clinical integration

- [x] 🇫🇮 Oura Ring - Cardiovascular AI
  - ⌚ Cardiovascular Age: Arterial stiffness + pulse wave velocity from PPG
  - ⌚ Oura Advisor (July 2024): AI-powered health coaching
  - 🎯 Project RESET: $25M Singapore program mapping heart disease

### Mental Health & Behavioral AI

- [x] 🇺🇸 Woebot
  - 💭 Rules-based + generative AI testing
  - 💭 CBT-based responses, dysfunctional thought recognition
  - 📊 Remarkable reductions in depression/anxiety
  - ⚠️ Users report generic/repetitive responses

- [x] 🇺🇸 Mindstrong
  - 📱 Smartphone usage pattern analysis (typing speed, app navigation)
  - 📊 90% accuracy detecting depression/anxiety (Nature Medicine 2023)
  - 🎯 Real-time monitoring, early warning

- [x] 🇺🇸 Hippocratic AI - Polaris
  - 🏥 Polaris 3.0: 99.38% clinical accuracy
  - 💰 $9/hour operating cost (vs $39/hour RN median)
  - 💬 Tested by 6,200+ nurses, 300+ doctors
  - 🤝 NVIDIA H100 GPUs, Universal Health Services deployment

---

## 📊 Tabular & Specialized Data Foundation Models

- [x] 🇩🇪 University of Freiburg
  - 📊 TabPFN v2: Bayesian approach, works "out of the box" on time-series
  - 💾 1M+ downloads, 5-10x more data than v1
  - 🔧 CAAFE: Automated feature engineering with LLMs
  - 🏛️ ELLIS unit Freiburg, OpenEuroLLM participation

---

## 📚 Infrastructure & Dataset Providers

- [x] 🇩🇪 LAION (Large-scale AI Open Network)
  - 📊 LAION-5B, LAION-400M, Re-LAION-5B
  - 🎨 Enabled Stable Diffusion, Imagen training
  - ⚖️ Won legal case on TDM exceptions (Sept 2024)
  - 🏛️ German nonprofit

---

## 📈 Summary Statistics

**Total Organizations Researched: 150+**

**By Category (with entries):**
- 🎨 Image Generation: 1 (Black Forest) + 4 multimodal
- 🎬 Video Generation: 11 major companies
  - US: 5 frontier (OpenAI, Runway, Google, Meta, Midjourney), 2 specialized (Pika, Luma)
  - Avatar/Enterprise: 3 (Synthesia, HeyGen, D-ID)
  - Asian: 1 (Kuaishou Kling)
- 🎙️ Audio/Speech: 16+ companies
  - Voice synthesis: 7 major (ElevenLabs, Microsoft, Meta, PlayAI, Cartesia, Descript, Resemble)
  - Music generation: 9+ (Suno, Udio, Stability AI, Meta MusicGen, Google, ByteDance, Adobe, Riffusion, Splash, AIVA, Boomy, Soundraw, Mubert)
- 🤖 Robotics/Embodied AI: 17+ companies/labs
  - US humanoid: 7 (Tesla, Figure AI, Physical Intelligence, Agility, NVIDIA GR00T, 1X, Sanctuary)
  - Chinese: 4 (UBTech, AgiBot, Huawei, Unitree)
  - Research: 6+ (Google DeepMind, Stanford, Berkeley, MIT, Toyota, CMU)
- 🧬 Scientific/Biological: 25+ organizations
  - Protein/Molecular: 12 (DeepMind, OpenProtein, EvolutionaryScale, Profluent, Genentech, Schrödinger, Exscientia, Recursion, Insitro, Tempus, Chai, UW)
  - Earth Science/Climate: 8 (Google DeepMind, NASA-IBM, NVIDIA, Huawei, ESA, Microsoft, etc.)
  - Materials Science: 3 (DeepMind GNoME, Meta UMA, Microsoft MatterGen)
- 🏥 Medical/Healthcare: 28+ organizations
  - Medical imaging: 12 (Google, Microsoft, Paige, Providence, Aignostics, PathAI, Aidoc, Tempus, IBM, Siemens, Stanford, Harvard)
  - Clinical decision support: 5 (OpenAI, Anthropic, Mistral, Meta, M42)
  - Wearables: 3 (Apple, Google Fitbit, Oura Ring)
  - Mental health: 3 (Woebot, Mindstrong, Hippocratic)
- 📊 Tabular Data: 1 (University of Freiburg)
- 📚 Infrastructure: 1 (LAION)

**Geographic Distribution:**
- 🇺🇸 United States: 80+ organizations (dominance in video, robotics, medical AI)
- 🇨🇳 China: 8+ (robotics leaders, music/weather AI)
- 🇬🇧 United Kingdom: 6 (DeepMind, Stability AI, Exscientia, Boston Dynamics HQ moves, D-ID, Patrick AI parent)
- 🇪🇺 Europe: 8+ (Black Forest, Synthesia, AIVA, Aignostics, ESA, Huawei Research)
- 🇦🇺 Australia: 1 (Splash Music)
- 🇯🇵 Japan: 2 (Soundraw, OpenProtein collaborations)
- 🇮🇱 Israel: 2 (D-ID, Aidoc)
- 🇫🇮 Finland: 1 (Oura Ring)

**Key Trends (2024-2025):**
1. **Video Gen Convergence:** <1 minute generation becoming standard, audio integration essential
2. **Voice AI Latency War:** 40ms (Cartesia Sonic), 75ms (ElevenLabs Flash) targets achieved
3. **Robotics Scale-Up:** Manufacturing ready, Tesla 5K, UBTech 500+, AgiBot mass production
4. **Multimodal Medical:** Image + genomic + clinical text integration (Tempus $200M initiative)
5. **Open Science Momentum:** AlphaFold 3 released, Boltz-2 open-source, Stable Audio open
6. **Chinese Acceleration:** UBTech swarms, AgiBot 1M datasets, Huawei Pangu scale
7. **Foundation Model Consolidation:** Smaller models (3B-7B) beating large models (55B+)
8. **Computational Efficiency:** 100,000x speedups (NeuralGCM), 5,000x (Aurora weather)

**Funding Highlights:**
- Largest valuations: Synthesia $4B, Runway $3B, Suno $500M, Pika $700M, ElevenLabs $3.3B
- Most funded: Tempus (IPO 2024), Recursion ($50M NVIDIA), Synthesia ($536M)
- Acquisitions: Meta acquired PlayAI (July 2025), Recursion acquired Exscientia (Nov 2024)

**Next Steps for Expansion:**
- Add more detailed model parameters and architecture specifications
- Include performance benchmarks and comparison tables
- Add regulatory approval status (FDA, CE, etc.)
- Include collaboration and partnership networks
- Add predicted 2026 developments
