#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include <signal.h>
#include <execinfo.h>
static void probe_crash(int sig){void*b[40];int n=backtrace(b,40);backtrace_symbols_fd(b,n,2);_exit(128+sig);}
int main(int argc,char**argv){@autoreleasepool{
 if(argc!=4){fprintf(stderr,"usage: run runtime.bundle plain.model audio.pcm\n");return 64;}
 setbuf(stdout,NULL);setbuf(stderr,NULL);signal(SIGSEGV,probe_crash);signal(SIGABRT,probe_crash);
 auto load_begin=std::chrono::steady_clock::now();
 void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h){fprintf(stderr,"%s\n",dlerror());return 1;}
 Class c=NSClassFromString(@"SAMICore"); Dl_info info;dladdr((void*)class_getMethodImplementation(c,sel_registerName("destroy")),&info);auto base=(uintptr_t)info.dli_fbase;
 fprintf(stderr,"loaded base=%p\n",(void*)base);
 void*manager=calloc(1,128);
 ((void*(*)(void*))(base+0x369918))(manager);
 fprintf(stderr,"manager constructed\n");
 std::string path=argv[2];
 int ret=((int(*)(void*,const std::string&))(base+0x36a038))(manager,path);
 fprintf(stderr,"model load ret=%d\n",ret);
 if(ret) _exit(2);
 void*executor=((void*(*)(void*))(base+0x36a36c))(manager);
 void*asr=calloc(1,80);((void*(*)(void*,void*))(base+0x368e78))(asr,executor);
 int init=((int(*)(void*))(base+0x368f44))(asr);std::string context="";
 int start=((int(*)(void*,const std::string&))(base+0x368f88))(asr,context);
 fprintf(stderr,"asr init=%d start=%d\n",init,start);
 fprintf(stderr,"load seconds=%.3f\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-load_begin).count());
 std::ifstream audio(argv[3],std::ios::binary);std::string pcm((std::istreambuf_iterator<char>(audio)),{});
 if(pcm.empty()){fprintf(stderr,"no audio\n");_exit(3);}
 auto begin=std::chrono::steady_clock::now();
 auto feed=(int(*)(void*,const std::string&,bool))(base+0x369218);
 auto get=(std::vector<std::string>(*)(void*,int*))(base+0x3693a4);
 for(size_t i=0;i<pcm.size();i+=3200){std::string chunk=pcm.substr(i,3200);ret=feed(asr,chunk,false);if(ret){fprintf(stderr,"feed error %d\n",ret);_exit(4);}int err=0;auto out=get(asr,&err);for(auto &t:out)if(!t.empty())printf("RESULT %s\n",t.c_str());}
 std::string empty;ret=feed(asr,empty,true);fprintf(stderr,"finish=%d\n",ret);
 for(int i=0;i<1;i++){int err=0;auto out=get(asr,&err);for(auto &t:out)if(!t.empty())printf("FINAL %s\n",t.c_str());}
 fprintf(stderr,"inference seconds=%.3f\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
 _exit(0);
}return 0;}
