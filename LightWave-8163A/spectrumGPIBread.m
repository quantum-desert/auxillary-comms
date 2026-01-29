%delete(instrfindall)
%g1 = serial('COM4');                  %Set COM Port
%status=get(g1,'status');
%g1.Terminator = 'CR/LF';
%set(g1,'InputBufferSize',10*20001);     %Creates enough space for OSA data
%fopen(g1);                              %Opens connection to the OSA
%idn=query(g1,'*IDN?');

%%%preinstall instrument control toolbox required

g1 = visadev("GPIB0::1::0::INSTR");

writeline(g1,'LDATC');
I0=readline(g1);
I1=str2num(I0); 
I1=I1(2:end);

writeline(g1,'WDATC');
w0=readline(g1);
w1=str2num(w0);
w1=w1(2:end);

figure
plot(w1,I1);

fid=fopen('intensity.txt','wt');
fprintf(fid,'%g\n',I1);
fclose(fid);

fid=fopen('lamda.txt','wt');
fprintf(fid,'%g\n',w1);
fclose(fid);

clear g1;