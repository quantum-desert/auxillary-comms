# Script to communicate with SRS760 and perform a range of functionality

# imports
import pyvisa as pyv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from functools import partial
import time
import binascii
import os
import keyboard
import asyncio
import logging
import tkinter as tk


################################# DEVICE SETUP #####################################

# constants
lf = "\n"
query = "?"
ident_key = "SR760"
num_bins = 400
disc = True # disconnect flag
goal = -57
num_samples = 4500 # 30 minutes of samples

# Configure logging
# logging.basicConfig(level=logging.DEBUG)

# command dict
# note: default trace g=0
global commands
commands = {
	"identify":"*IDN?\n",
	"center_f":"CTRF",
	"span":"SPAN",
	"start":"STRT",
	"avg_on":"AVGO1"+lf,
	"avg_off":"AVGO0"+lf,
	"marker_x":"MRKX?0"+lf,
	"marker_y":"MRKY?0"+lf,
	"marker_w":"MRKW0,",
	"marker_status":"MRKR0,",
	"marker_mode":"MRKM0,",
	"marker_bin":"MBIN0,",
	"top_ref_set":"TREF0,",
	"bot_ref_set":"BREF0,",
	"y_division":"YDIV0,",
	"err_chk":"ERRS?"+lf,
	"num_avg":"NAVG",
	"avg_type":"AVGT",
	"get_x":"SPEC?0,",
	"set_ydiv":"YDIV0,",
	"measure_spectrum":"SPEC?0"+lf,
	"avg_complete?":"FFTS?4",
}
# settings
span = 14 # 3,125kHz -> if change refer to "SPAN (?)" CMD dictionary in SRS760 manual

spant_dict = {14:3125,19:100000}

global params
params = {
	"center_f": 4600,
	"span":span,
	"spot":2,
	"mkr_on":1,
	"mkr_seek_max":0,
	"mkr_bin_center":int(num_bins/2),
	"tref":-62,
	"bref":-94,
	"ydiv":4,
	"peak_hold":2,
	"RMS":0,
	"max_avg_num":32000,
	"y_div":10,
	"avg_num":1000,

}
GPIB_addr = 10 # set on SRS settings front panel
avg_status = 0 # off

# GPIB setup
rm = pyv.ResourceManager()
#print(rm.list_resources())

def connect():

	SRS760 = rm.open_resource('GPIB0::'+str(GPIB_addr)+'::INSTR')

	# attempt cxn
	print("Attempting connection...")
	SRS760.write(commands["identify"])
	result = SRS760.read()
	if(ident_key in result):
		print("Connection Established to "+ident_key+"!")
		# clear output buffer
		SRS760.clear()

		# feed object back to program
		return SRS760
	else:
		print("Connection unsuccessful")


	return None

	# if(device.read_stb() & 0x10):
	# 	result = device.read()
	# 	print(result)

def setup(device,params):
	cmd = ""

	# set center frequency
	cmd = commands["center_f"]+str(params["center_f"])+lf
	device.write(cmd)

	# set span
	cmd = commands['span']+str(params['span'])+lf
	device.write(cmd)

	# # turn off average
	# cmd = commands['avg_off']
	# device.write(cmd)

	# turn marker on
	cmd = commands['marker_status']+str(params['mkr_on'])+lf
	device.write(cmd)

	# set marker width
	cmd = commands['marker_w']+str(params['spot'])+lf
	device.write(cmd)

	# set marker mode
	cmd = commands['marker_mode']+str(params['mkr_seek_max'])+lf
	device.write(cmd)

	# set marker position
	cmd = commands['marker_bin']+str(params['mkr_bin_center'])+lf
	device.write(cmd)

	# set y division
	cmd = commands['y_division']+str(params['ydiv'])+lf
	device.write(cmd)

	# set top ref
	cmd = commands['top_ref_set']+str(params['tref'])+lf
	device.write(cmd)

	# set bottom ref
	cmd = commands['bot_ref_set']+str(params['bref'])+lf
	device.write(cmd)

	# set num. average
	# cmd = commands['num_avg']+str(params['avg_num'])+lf
	# device.write(cmd)

	# set average type
	# cmd = commands['avg_type']+str(params['peak_hold'])+lf
	# cmd = commands['avg_type']+str(params['RMS'])+lf

	device.write(cmd)

	# # turn on average
	# cmd = commands['avg_on']
	# device.write(cmd)

	# start measurement (last thing)
	cmd = commands['start']+lf
	device.write(cmd)

	# wait until average is complete
	# cmd = commands['avg_complete?']+lf
	# print('Waiting for average to complete...')
	# device.write(cmd)
	# complete = int(device.read().strip('\n'))
	# while(not complete):
	# 	device.write(cmd) # query again
	# 	complete = int(device.read().strip('\n'))
	# print('Average complete.')


def disconnect(device):
	device.close()



#################################################################################

################################# DATA ANIMATION ################################

# init fxn
def init(ln,ln2):
	ln.set_data([],[])
	ln2.set_data([],[])

	return ln,ln2

# # data update fxn
# def update(frame):
# 	f.append(frame)


## POLLING
def poll(frame,device,x,A,mA,ax,ln,ln2):
	# time iteratons
	# global prev_time
	# current = time.time()
	# diff = current - prev_time
	# prev_time = time.time()
	# print("Time between plots:"+str(diff*1e3)+"ms")
	# **************time between iterations (350-400 ms)******************

	# poll_command = commands['marker_y'] # poll marker position
	poll_command = commands['get_x']+str(params['mkr_bin_center'])+lf # directly poll middle bin
	device.write(poll_command)
	data = float(device.read().strip('\n'))

	# check for errors
	# err_cmd = commands['err_chk']
	# device.write(err_cmd)
	# err_byte = device.read().strip('\n').encode()
	# hex_value = binascii.hexlify(err_byte)
	# print("Hexadecimal value:", hex_value.decode())

	x.append(frame)
	A.append(data)
	mA.append(np.mean(A[max(-1*len(A),-20):-1]))
	ax.set_xlim([max(0,frame-num_samples),frame+1])
	# force canvas update
	ax.figure.canvas.draw()
	ln.set_data(x,A)
	ln2.set_data(x,mA)
	print("Peak: "+str(A[-1]))
	print("Moving Avg: "+str(mA[-1])+"\n")
	return ln,ln2

# Generate dataset
def get_data(device):


	# get center f
	center_f_query_cmd = commands['center_f']+'?'+lf
	device.write(center_f_query_cmd)
	center_f = float(device.read().strip('\n'))
	
	# get span
	span_query_cmd = commands['span']+'?'+lf
	device.write(span_query_cmd)
	span = spant_dict[int(device.read().strip('\n'))]

	# get trace data (dbV)
	get_data_cmd = commands['measure_spectrum']
	device.write(get_data_cmd)
	trace_data = device.read().strip('\n')
	trace_data = trace_data.split(',')
	trace_data.pop(-1)
	i=0
	for t in trace_data:
		trace_data[i] = float(t)
		i=i+1


	# generate span data
	lo = center_f - span/2
	hi = center_f + span/2
	span=np.linspace(lo,hi,len(trace_data))


	return [span, trace_data]

def get_data_fast(device):
	get_data_cmd = commands['measure_spectrum']
	device.write(get_data_cmd)
	trace_data = device.read().strip('\n')
	trace_data = trace_data.split(',')
	trace_data.pop(-1)
	trace_data = [float(t) for t in trace_data]
	# i=0
	# for t in trace_data:
	# 	trace_data[i] = float(t)
	# 	i=i+1

	return trace_data

	
# Generate frames infinitely
def data_gen():
    frame = 0
    while True:
        yield frame
        frame += 1  # Increase frame count (can adjust step as needed)

def plot_setup(fig,ax,x,A,mA):
	ax.axhline(goal,linestyle='--',color='r',linewidth=2,label='Goal')
	ln, = plt.plot(x,A,'b',animated=True,linewidth=2,label='Signal')
	ln2, = plt.plot(x,mA,'g',animated=True,linewidth=2,label='Moving Avg.')
	# ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
	ax.set_ylim([params['bref'],params['tref']]) # dBV
	ax.set_xlim([0,10])
	ax.set_ylabel('Signal (dBV)')
	ax.set_xlabel('Sample #')
	ax.set_title('Peak Tracking')
	ax.grid(True)
	ax.legend(loc='center left')
	return ln,ln2

def plot_setup_noanim(fig,ax,f):
	ln, = plt.plot([],[],'b',linewidth=2,label='Signal')
	ax.set_xlim([min(f), max(f)])
	ax.set_ylim([params['bref'],params['tref']]) # dBV
	ax.set_ylabel('Signal (dBV)',fontsize=16)
	ax.set_xlabel('Frequency (kHz)',fontsize=16)
	ax.set_title('Spectrum',fontsize=16)
	ax.grid(True)
	return ln

def save_to_file(x,y,name,loc, header):
	# saves assumes two column data format
	# stack data
	stacked_data = np.column_stack((x,y))

	# account for overwrite
	if(os.path.exists(loc+name)):
		name = name.split('.txt')
		update = str(int(name[0][-1])+1)
		name = name[0][0:-2]+'_'+update+'.txt'
	name=loc+name
	print('Saving data...')
	np.savetxt(name,stacked_data,fmt='%f',delimiter=',',header=header)
	print('Data saved.')

def update(frame,device,ln,f,S,data_queue,ax,text):
	# logging.debug(f"Queue size: {data_queue.qsize()}")
	if not data_queue.empty():
		# logging.debug("get data from queue")
		S = data_queue.get_nowait()
	# else:
		# logging.debug("Dont get data from queue")
	ln.set_data(f,S)
	max_v = np.max(S)
	max_i = np.argmax(S)
	text.set_text(f"Max Peak: {round(max(S),2)} [dBV]")
	text.set_position((f[max_i],max_v))
	return text,ln

def init(ln):
    ln.set_data([], [])
    return ln,

async def poll_device_async(device, poll_interval, data_queue):
    """
    Asynchronously polls the device for data and puts it into a queue.

    Args:
        device: The device object to communicate with.
        poll_interval: Time in seconds between polls.
        data_queue: An asyncio.Queue to hold the data.
    """
    try:
        while True:
            # Fetch data from the device
            get_data_cmd = commands['measure_spectrum']
            device.write(get_data_cmd)
            trace_data = device.read().strip('\n').split(',')
            trace_data = [float(t) for t in trace_data[:-1]]

            # Put the data into the queue for processing
            await data_queue.put(trace_data)

            # Wait for the next poll
            await asyncio.sleep(poll_interval)
    except Exception as e:
        print(f"Error while polling device: {e}")


#################################################################################

################################# Execution #####################################
# execution
def main():
	# setup plots
	fig,ax = plt.subplots()
	# x,A,mA = [],[],[]
	# ln,ln2 = plot_setup(fig,ax,x,A,mA)

	# # plots for static reproduction
	# half_span = int(spant_dict[params['span']]/2)
	# f=np.linspace(params['center_f']-half_span,params['center_f']+half_span,num_bins)
	# S = -100.*np.ones(np.shape(f)) # initialize


	# root = tk.Tk()
	# screen_width = root.winfo_screenwidth()
	# screen_height = root.winfo_screenheight()
	# root.destroy()

	# fig, ax = plt.subplots(figsize=(screen_width/100, screen_height/100))  # Adjust dpi if needed
	# ln = plot_setup_noanim(fig, ax, f)
	# text = ax.text(0, 0, "", fontsize=16,fontweight="bold",color="r")

	

	# connect to device
	device = connect()
	if(device is None):
		print("Error connecting")
		exit()

	# data_queue = asyncio.Queue()


	# # perform device setup
	# setup(device, params)

	# # start async polling function
	# asyncio.create_task(poll_device_async(device,0.01,data_queue))
	# logging.info("Created polling task")

	# # while True:
	# # 	active_tasks = asyncio.all_tasks()
	# # 	logging.debug(f"Active tasks: {len(active_tasks)}")
	# # 	logging.debug(f"Queue size: {data_queue.qsize()}")

	# # 	await asyncio.sleep(1)


	# # update fxn cor animation
	# # def update(frame):
	# # 	if not data_queue.empty():
	# # 		data = data_queue.get_nowait()
	# # 		ln.set_data(f,data)
	# # 	return ln,



	# # PLOTTING F TRACE POLLING #
	# print("**** Beginning frequency trace poll... ***")

	# # ani = FuncAnimation(fig,update,blit=True,interval=100)

	# # animation
	# ani=FuncAnimation(fig,update,blit=True,interval=10,fargs=(device,ln,f,S,data_queue,ax,text))
	# print("FuncAnimation has been set up.")



	# # show plot in non-blocking way
	# plt.show(block=False)

	# # continue running async loop
	# while plt.fignum_exists(fig.number):
	# 	plt.pause(0.05)
	# 	await asyncio.sleep(0.05)


	# # disconnect if flag set
	# if(disc):
	# 	disconnect(device)
	# 	print("SRS760 Connection closed")





	# PLOTTING SINGLE F TRACE #

	# grab one dataset
	[f,A] = get_data(device)

 	# export data for processing in MATLAB
	save_loc = os.getcwd()+r'\\'
	print(save_loc)
	header = 'f [Hz], A [dBV]\n'
	filename = 'f_trace_TIA_S2_PSB.txt'
	save_to_file(f,A,filename,save_loc,header)
	# NB=451 # defien
	# save_loc = r'C:/Users/liamw/OneDrive - Umich/GraduateSchool/UM/QE_LAB/MAC/data/7-8-2025-C1/'
	# header = 'f [Hz], A [dBV]\nNB='+str(NB)
	# save_to_file(f,A,'f_trace_present_NB0_7_8.txt',save_loc,header)

	# plot single dataset	
	ax.plot(f,A)
	ax.set_title('Frequency Domain Trace')

	disconnect(device)
	print("SRS760 Connection closed")

	# # poll
	# init_w_arg = partial(init,ln=ln,ln2=ln2)
	# update_w_arg = partial(poll, device=device,x=x,A=A,mA=mA,ax=ax,ln=ln,ln2=ln2)
	# ani = animation.FuncAnimation(fig, update_w_arg, frames=data_gen,
                              # init_func=init_w_arg, blit=True)

	# show plot
	#plt.show()

	


if __name__ == "__main__":
	main()
# asyncio.run(main())
#################################################################################	