// The RL training executable never initializes networking or audio. These
// link-time stubs keep the legacy type surface available without loading the
// x86-only SDL2_net, Ogg, or Vorbis binaries, or the system OpenAL framework.

#include <SDL_net.h>

#include <Wrappers/openal.h>

#include <vorbis/vorbisfile.h>

extern "C" {

int SDLCALL SDLNet_Init(void) { return 0; }
void SDLCALL SDLNet_Quit(void) {}
int SDLCALL SDLNet_ResolveHost(IPaddress*, const char*, Uint16) { return -1; }
TCPsocket SDLCALL SDLNet_TCP_Open(IPaddress*) { return nullptr; }
TCPsocket SDLCALL SDLNet_TCP_Accept(TCPsocket) { return nullptr; }
int SDLCALL SDLNet_TCP_Send(TCPsocket, const void*, int) { return -1; }
int SDLCALL SDLNet_TCP_Recv(TCPsocket, void*, int) { return -1; }
void SDLCALL SDLNet_TCP_Close(TCPsocket) {}
SDLNet_SocketSet SDLCALL SDLNet_AllocSocketSet(int) { return nullptr; }
int SDLCALL SDLNet_AddSocket(SDLNet_SocketSet, SDLNet_GenericSocket) { return -1; }
int SDLCALL SDLNet_DelSocket(SDLNet_SocketSet, SDLNet_GenericSocket) { return -1; }
int SDLCALL SDLNet_CheckSockets(SDLNet_SocketSet, Uint32) { return 0; }
void SDLCALL SDLNet_FreeSocketSet(SDLNet_SocketSet) {}
const char* SDLCALL SDLNet_GetError(void) { return "networking disabled in RL headless build"; }

static ALuint next_audio_handle = 1;

void AL_APIENTRY alBufferData(ALuint, ALenum, const ALvoid*, ALsizei, ALsizei) {}
void AL_APIENTRY alDeleteBuffers(ALsizei, const ALuint*) {}
void AL_APIENTRY alDeleteSources(ALsizei, const ALuint*) {}
void AL_APIENTRY alDistanceModel(ALenum) {}
void AL_APIENTRY alGenBuffers(ALsizei count, ALuint* buffers) {
    for (ALsizei i = 0; i < count; ++i) buffers[i] = next_audio_handle++;
}
void AL_APIENTRY alGenSources(ALsizei count, ALuint* sources) {
    for (ALsizei i = 0; i < count; ++i) sources[i] = next_audio_handle++;
}
ALenum AL_APIENTRY alGetError(void) { return AL_NO_ERROR; }
void AL_APIENTRY alGetSourcei(ALuint, ALenum, ALint* value) {
    if (value) *value = AL_STOPPED;
}
const ALchar* AL_APIENTRY alGetString(ALenum) { return "RL headless audio disabled"; }
ALboolean AL_APIENTRY alIsBuffer(ALuint buffer) { return buffer ? AL_TRUE : AL_FALSE; }
ALboolean AL_APIENTRY alIsSource(ALuint source) { return source ? AL_TRUE : AL_FALSE; }
void AL_APIENTRY alListener3f(ALenum, ALfloat, ALfloat, ALfloat) {}
void AL_APIENTRY alListenerf(ALenum, ALfloat) {}
void AL_APIENTRY alListenerfv(ALenum, const ALfloat*) {}
void AL_APIENTRY alSource3f(ALuint, ALenum, ALfloat, ALfloat, ALfloat) {}
void AL_APIENTRY alSourcePlay(ALuint) {}
void AL_APIENTRY alSourceQueueBuffers(ALuint, ALsizei, const ALuint*) {}
void AL_APIENTRY alSourceStop(ALuint) {}
void AL_APIENTRY alSourceUnqueueBuffers(ALuint, ALsizei count, ALuint* buffers) {
    for (ALsizei i = 0; buffers && i < count; ++i) buffers[i] = 0;
}
void AL_APIENTRY alSourcef(ALuint, ALenum, ALfloat) {}
void AL_APIENTRY alSourcei(ALuint, ALenum, ALint) {}

ALCdevice* ALC_APIENTRY alcOpenDevice(const ALCchar*) {
    return reinterpret_cast<ALCdevice*>(1);
}
ALCboolean ALC_APIENTRY alcCloseDevice(ALCdevice*) { return ALC_TRUE; }
ALCcontext* ALC_APIENTRY alcCreateContext(ALCdevice*, const ALCint*) {
    return reinterpret_cast<ALCcontext*>(1);
}
ALCboolean ALC_APIENTRY alcMakeContextCurrent(ALCcontext*) { return ALC_TRUE; }
void ALC_APIENTRY alcDestroyContext(ALCcontext*) {}
ALCenum ALC_APIENTRY alcGetError(ALCdevice*) { return ALC_NO_ERROR; }
const ALCchar* ALC_APIENTRY alcGetString(ALCdevice*, ALCenum) { return "RL headless audio disabled"; }
ALCboolean ALC_APIENTRY alcIsExtensionPresent(ALCdevice*, const ALCchar*) { return ALC_FALSE; }

int ov_clear(OggVorbis_File*) { return 0; }
int ov_open(FILE*, OggVorbis_File*, const char*, long) { return OV_EIMPL; }
ogg_int64_t ov_pcm_total(OggVorbis_File*, int) { return 0; }
int ov_pcm_seek(OggVorbis_File*, ogg_int64_t) { return OV_EIMPL; }
int ov_raw_seek_lap(OggVorbis_File*, ogg_int64_t) { return OV_EIMPL; }
ogg_int64_t ov_pcm_tell(OggVorbis_File*) { return 0; }
vorbis_info* ov_info(OggVorbis_File*, int) { return nullptr; }
vorbis_comment* ov_comment(OggVorbis_File*, int) { return nullptr; }
long ov_read(OggVorbis_File*, char*, int, int, int, int, int*) { return 0; }

}  // extern "C"
