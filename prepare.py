import struct,pathlib,hashlib
root=pathlib.Path(__file__).resolve().parent
src=pathlib.Path('/Library/Input Methods/DoubaoIme.app/Contents/MacOS/DoubaoIme').read_bytes()
assert hashlib.sha256(src).hexdigest() == '5b48748fa6394811cba0f8c05905ffebb9a2cd02faf6d47b26deb3d9b86247e0', 'Unsupported app binary version'
n=struct.unpack_from('>I',src,4)[0]
for i in range(n):
 c,sub,off,size,align=struct.unpack_from('>IIIII',src,8+20*i)
 if c==0x100000c: b=bytearray(src[off:off+size]);break
ncmd,sz=struct.unpack_from('<II',b,16);pos=32;cmds=[]
for i in range(ncmd):
 c,s=struct.unpack_from('<II',b,pos);cmd=bytearray(b[pos:pos+s]);pos+=s
 if c in (0xe,0x80000028,0x1d):continue
 # Preserve segment indexes used by dyld rebase/bind opcodes.
 if c==0x19 and cmd[8:24].rstrip(b'\0')==b'__PAGEZERO':struct.pack_into('<QQ',cmd,24,0x100000000,0);cmd[8:24]=b'__EMPTY'+b'\0'*9
 if c==0xc and b'libDoubaoImeDylib.dylib' in cmd:
  name=b'@loader_path/helper.dylib\0';cmd[24:]=name+b'\0'*(len(cmd)-24-len(name))
 cmds.append(cmd)
raw=b''.join(cmds);b[32:32+sz]=raw+b'\0'*(sz-len(raw))
struct.pack_into('<III',b,12,8,len(cmds),len(raw))
flags=struct.unpack_from('<I',b,24)[0];struct.pack_into('<I',b,24,flags&~0x200000)
for address in (0x28be5c,0x29de08,0x29df3c):struct.pack_into('<I',b,address,0xd65f03c0)
(root/'runtime.bundle').write_bytes(b)
