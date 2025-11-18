#Execution
#############################################################################################################
#Use: --- nohup python postprocess.py > postprocess 2>&1 < /dev/null &  --- to run this script !
#############################################################################################################

#Libraries
#####################################################################################
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.colors as clr
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import subprocess
import os
from pylab import *
from scipy.interpolate import griddata
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module=".*fluidfoam.*")
import os
import sys
from contextlib import redirect_stdout
with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull):
        from fluidfoam import readmesh, readvector, readscalar, readfield
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import math
import time
#####################################################################################

#Timer on 
starttime = time.time()

#Parameters
###########
#Simulation 
sim='name'
W=2;
H=0.08;
beta=8;
g=9.81;
Reb=10000;
nu=1e-6;
rho=1000;
R=4.5e-4;
a0=0.003;

#Interpolation
sedfoam = 0;
timenames=['180'];
window = 30;
inst = 0;

#Visualization
colors = ["white", "#f0e442", "#7d3f00"]  
custom_colormap = LinearSegmentedColormap.from_list("wyb", colors)
ccolormap = custom_colormap;
vcolormap = plt.colormaps['PuOr_r']
if sedfoam == 0:
   scolormap = plt.colormaps['Purples'] 
else:
   scolormap = 'inferno';
ecolormap = plt.colormaps['Reds']
vcolormap = 'seismic';
seismic = plt.cm.seismic
tkecolormap = LinearSegmentedColormap.from_list('seismic_positive', seismic(np.linspace(0.5, 1, 256)))

#Statistics
flagssurf=0;
flagasurf=0;
flagusurf=0;
flagresurf=0;
flagtkesurf=0;
flagvortsurf=0;
flagqcritsurf=0; 
flagpsurf=0;
flagpasurf=0;
flagstreamsurf=0;
flagksurf=0;
flagscenter=0;
flagacenter=0;
flagucenter=0;
flagrecenter=0;
flagtkecenter=0;
flagvortcenter=0;
flagqcritcenter=0;
flagpcenter=0;
flagpacenter=0;
flagstreamcenter=0;
flagkcenter=0;
flagshearcenter=0;
flagep=0;
flagqout=0;
flagstrans=0;
flagatrans=0;
flagutrans=0;
flagstreamtrans=0;
flagsclin=0;
flagaclin=0;
flaguclin=0;
flagnutsurf=0; 
flagnutcenter=0;
flagcompeddviscsurf=0;
flagcompeddvisccenter=0;
flagcompeddvisctrans=0;
flagkhwave=0; 
flagriverlake3Dinterface=0;
flagmodel=0; 

#Grid
refinementfactor=2;
Lxt=16;
Lyt=12;
Lzt=1.76;
Lxb=12;
Lyb=12;
Lzb=1.76;
Nx=275;
Ny=150;
Nz=45;
Nxb = int(Nx * Lxb / Lxt * 2 ** refinementfactor);
Nyb = int(Ny * Lyb / Lyt * 2 ** refinementfactor);
Nzb = int(Nz * Lzb / Lzt * 2 ** refinementfactor);
Lxc=4;
Lyc=W;
Lzc=H;
Nx=275;
Ny=150;
Nz=45;
Nxc = int(Nx * Lxc / Lxt * 2 ** refinementfactor);
Nyc = int(Ny * Lyc / Lyt * 2 ** refinementfactor);
Nzc = int(Nz * Lzc / Lzt * 2 ** refinementfactor);

#Surface-plane
Zvert_surf=0;
dz_surf=0.05;
ngridx_surf = Nxb;
ngridy_surf = Nyb;
yinterpmin_surf = -Lyb/6;
yinterpmax_surf = +Lyb/6;
xinterpmin_surf = 0;
xinterpmax_surf = Lxb/2;
tkefiltercutoff = 0.1;

#Center-plane
Yplane_center=0;
dy_center=0.05;
ngridx_center = Nxb;
ngridz_center = Nzb;
xinterpmin_center = 0;
xinterpmax_center = Lxb/2;
zinterpmin_center = -Lzb;
zinterpmax_center = 0;
Dcr=-(H+tan(beta/180*np.pi)*xinterpmax_center);

#Transversal-plane
Xplane_trans=0;
Dtr=-Xplane_trans*tan(beta/180*pi)-H;
dx_trans=0.05;
ngridy_trans = Nyb;
ngridz_trans = Nzb;
yinterpmin_trans = -Lyb/6;
yinterpmax_trans = +Lyb/6;
zinterpmin_trans = -Lzb
zinterpmax_trans = 0

#Inclined-plane
Zvert_clin=H/2;
dz_clin=0.005;
ngridx_clin = Nxb;
ngridy_clin = Nyb;
xinterpmin_clin = 0;
xinterpmax_clin = Lxb;
yinterpmin_clin = -Lyb/4;
yinterpmax_clin = +Lyb/4;

#Ep coefficient
ngridy_ep = Nyb;
ngridz_ep = Nzb;
yinterpmin_ep = -Lyb/2;
yinterpmax_ep = +Lyb/2;
zinterpmin_ep = -Lzb;
zinterpmax_ep = 0;
dx_ep=0.05;
step=5;
Xplane_channel =-0.5
dx_channel=0.05
ngridy_channel = Nyc
ngridz_channel = Nzc*10
yinterpmin_channel = -W/2
yinterpmax_channel = +W/2
zinterpmin_channel = -H
zinterpmax_channel = 0 

#Check
print("#" * 100)
print(f"{'Input parameters check':^100}")
print("#" * 100)

print(f"Grid points in tank: x = {Nxb:<5}  -  y = {Nyb:<5}  -  z = {Nzb:<5}")
print(f"Interpolation points surfaceplane: x = {ngridx_surf:<5}  -  y = {ngridy_surf:<5}")
print("-" * 100)

print("Surface plane statistics:")
print(f"  Salinity/Sediment : {flagssurf}/{flagasurf}")
print(f"  Velocity          : {flagusurf}")
print(f"  Reynolds stress   : {flagresurf}")
print(f"  Vorticity         : {flagvortsurf}")
print(f"  TKE               : {flagtkesurf}")
print(f"  Pressure / pa     : {flagpsurf} / {flagpasurf}")
print(f"  Sub-grid nut      : {flagnutsurf}")
print(f"  Computed nut      : {flagcompeddviscsurf}")
print(f"  Q-criterion       : {flagqcritsurf}")
print(f"  Sub-grid k        : {flagksurf}")
print(f"  Streamlines       : {flagstreamsurf}")
print("-" * 100)

print("Center plane statistics:")
print(f"  Salinity/Sediment : {flagscenter}/{flagacenter}")
print(f"  Velocity          : {flagucenter}")
print(f"  Reynolds stress   : {flagrecenter}")
print(f"  Vorticity         : {flagvortcenter}")
print(f"  TKE               : {flagtkecenter}")
print(f"  Pressure / pa     : {flagpcenter} / {flagpacenter}")
print(f"  Sub-grid nut      : {flagnutcenter}")
print(f"  Computed nut      : {flagcompeddvisccenter}")
print(f"  Q-criterion       : {flagqcritcenter}")
print(f"  Sub-grid k        : {flagkcenter}")
print(f"  Streamlines       : {flagstreamcenter}")
print(f"  Shear stress      : {flagshearcenter}")
print("-" * 100)

print("Transversal plane statistics:")
print(f"  Salinity/Sediment : {flagstrans}/{flagatrans}")
print(f"  Velocity          : {flagutrans}")
print(f"  Streamlines       : {flagstreamtrans}")
print(f"  Computed nut      : {flagcompeddvisctrans}")
print("-" * 100)

print("Inclined plane statistics:")
print(f"  Salinity/Sediment : {flagsclin}/{flagaclin}")
print(f"  Velocity          : {flaguclin}")
print("-" * 100)

print("Plunge statistics:")
print(f"  xp-99R% - Min da/dx   : {flagssurf}/{flagasurf}")
print(f"  Plume height hc(x)    : {flagucenter}")
print(f"  Ep(x)                 : {flagep}")
print(f"  Mass conservation     : {flagqout}")
print(f"  KH wave profile       : {flagkhwave}")
print(f"  River-lake 3D interface: {flagriverlake3Dinterface}")
print(f"  Analytical model      : {flagmodel}")
print("#" * 100)

#Reading
print(f"############################################################################################")
print(f"Reading fields")
print(f"############################################################################################")
##########################################
sol = '../../'

x, y, z = readmesh(sol)

for timename in timenames:
   if sedfoam==1:
     if inst==1:
       U = readvector(sol, timename, 'U.b')
       alpha = readscalar(sol, timename, 'alpha.a')
       filenameCurl = os.path.join(sol, timename, 'vorticity')
       if os.path.isfile(filenameCurl):
          CURL = readvector(sol, timename, 'vorticity')
       else:
          flagvortsurf=0
          flagvortcenter=0
       filenameQ = os.path.join(sol, timename, 'Q')
       if os.path.isfile(filenameQ):
          Qcrit = readscalar(sol, timename, 'Q')
       else:
          flagqcritsurf=0
          flagqcritcenter=0
       p = readscalar(sol, timename, 'p_rbgh')
       pa = readscalar(sol, timename, 'pa')
     else:
       U = readvector(sol, timename, 'UbMeanF')
       alpha=readscalar(sol, timename, 'alpha_aMean')
       tensor=readfield(sol, timename, 'UbPrime2MeanF')
       filenameCurl = os.path.join(sol, timename, f'vorticityMean_w{window}vorticity')
       if os.path.isfile(filenameCurl):
          CURL = readvector(sol, timename, f'vorticityMean_w{window}vorticity')
       else:
          flagvortsurf=0
          flagvortcenter=0
       filenameQ = os.path.join(sol, timename, f'QMean_w{window}Q')
       if os.path.isfile(filenameQ):
          Qcrit = readscalar(sol, timename, f'QMean_w{window}Q')
       else:
          flagqcritsurf=0
          flagqcritcenter=0
       print(f"Tensor shape: {tensor.shape}")
       Rxy = tensor[1]
       Rxz = tensor[2]
       Rxx = tensor[0]
       Ryy = tensor[4]
       Ryz = tensor[5]
       Rzz = tensor[8]
       TKE = Rxx + Ryy + Rzz  
   else:
     if inst==1:
       U = readvector(sol, timename, 'U')
       T = readscalar(sol, timename, 'T')
       filenameCurl = os.path.join(sol, timename, 'vorticity')
       if os.path.isfile(filenameCurl):
          CURL = readvector(sol, timename, 'vorticity')
       else:
          flagvortsurf=0
          flagvortcenter=0
       filenameQ = os.path.join(sol, timename, 'Q')
       if os.path.isfile(filenameQ):
          Qcrit = readscalar(sol, timename, 'Q')
       else:
          flagqcritsurf=0
          flagqcritcenter=0
       p = readscalar(sol, timename, 'p_rgh')
       nut = readscalar(sol, timename, 'nut')
       k = readscalar(sol, timename, 'k')
     else:
       U = readvector(sol, timename, f'UMean_w{window}U')
       T = readscalar(sol, timename, f'TMean_w{window}T')
       tensor=readfield(sol, timename, f'UPrime2Mean_w{window}U')
       filenameCurl = os.path.join(sol, timename, f'vorticityMean_w{window}vorticity')
       if os.path.isfile(filenameCurl):
          CURL = readvector(sol, timename, f'vorticityMean_w{window}vorticity')
       else:
          flagvortsurf=0
          flagvortcenter=0
       filenameQ = os.path.join(sol, timename, f'QMean_w{window}Q')
       if os.path.isfile(filenameQ):
          Qcrit = readscalar(sol, timename, f'QMean_w{window}Q')
       else:
          flagqcritsurf=0
          flagqcritcenter=0
       print(f"Tensor shape: {tensor.shape}")
       Rxy = tensor[1]
       Rxz = tensor[2]
       Rxx = tensor[0]
       Ryy = tensor[3]
       Ryz = tensor[4]
       Rzz = tensor[5]
       TKE = Rxx + Ryy + Rzz 
       filenameGrad = os.path.join(sol, timename, f'grad(UMean_w{window}U)')
       if os.path.isfile(filenameGrad):
          gradUmtensor = readfield(sol, timename, f'grad(UMean_w{window}U)')
          print(f"Gradient tensor shape: {gradUmtensor.shape}")
          Umxy = gradUmtensor[1]
          Umxz = gradUmtensor[2]
          Umxx = gradUmtensor[0]
          Umyy = gradUmtensor[3]
          Umzz = gradUmtensor[5]
       else:
          flagcompeddviscsurf=0
          flagcompeddvisccenter=0
          flagcompeddvisctrans=0
       
   #Postprocessing
   print(f"############################################################################################")
   print(f"Start postprocessing for time t={timename}s")
   print(f"############################################################################################")
   ##########################################

   #Plunging point based on 99%R criterion !
   if flagssurf == 1 and sedfoam==0:
     print(f"############################################################################################")
     print(f"Surface salinity contourmap !")
     print(f"############################################################################################")
     print(f"Plunging point based on 99%R criterion !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     T_i = griddata((x[Iplane], y[Iplane]), T[Iplane], (xinterp, yinterp), method='linear')

     for m in range (0,len(T_i[ngridy_surf//2])):
       if T_i[ngridy_surf//2][m]<=0.99*R:
          xp=xinterp[0][m]
          break
     print(f"######################################################################################################################")
     print(f'Plunging point position based on 99%R criterion for time={timename} x_p={xp:.5f} m')
     print(f'Plunging point position based on 99%R criterion for time={timename} normalized by the channel width x_p/W={(xp/W):.5f}')
     print(f"######################################################################################################################")

     #Surface salinity map !
     fig = plt.figure(figsize=(6, 4),dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     plt.contourf(xinterp/W, yinterp/W, T_i/R, cmap=scolormap, levels=np.linspace(0, np.max(T_i/R), 100), vmin=0, vmax=1)
     cbar = plt.colorbar()
     plt.contour(xinterp/W, yinterp/W, T_i/R, levels=[0.99], colors='red', linewidths=2)
     plt.axvline(x=xp/W, color='red', linestyle='--', linewidth=1.5)
     plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
     plt.plot(xp / W, 0, marker='*', color='red', markersize=10)
     ax=plt.gca()
     ax_top=ax.secondary_xaxis('top')
     ax_top.set_xticks([xp/W])
     ax_top.set_xticklabels(['$x_p$'],fontsize=24,color='black')
     if inst==0:
        cbar.set_label('$\\overline{R}$/$\\overline{{R}_{0}} \\%$', fontsize=28)
     else:
        cbar.set_label('${R}$/$\\overline{{R}_{0}}$ \\%', fontsize=28)
     cbar.ax.tick_params(labelsize=24)
     plt.tick_params(axis='x', labelsize=24) 
     plt.tick_params(axis='y', labelsize=24)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     cbar.set_ticks(np.linspace(0, 1, 6))
     def percentage(x, pos):
         return f'{x*100:.0f}%'  
     cbar.ax.yaxis.set_major_formatter(FuncFormatter(percentage))
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$y/W$', fontsize=28)
     filename='-surfaceplane-salinity-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches='tight')
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Surface salinity streamwise profile at y=0 !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(xinterp[int(ngridy_surf/2),1:], columns=['x']).to_csv(csvname, index=False)
     csvname ='-R-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(T_i[int(ngridy_surf/2),1:]/R, columns=['R']).to_csv(csvname, index=False)
     plt.plot(xinterp[int(ngridy_surf/2),1:]/W, T_i[int(ngridy_surf/2),1:]/R, c='black')
     plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
     plt.xlabel('$x/W$', fontsize=12)
     if inst==0:
        plt.ylabel('$\\overline{R}$/${R}_{0}$', fontsize=12)
     else:
        plt.ylabel('${R}$/${R}_{0}$', fontsize=12)
     filename='-surfaceplane-streamwise-salinity-profile-at-y=0.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Surface salinity streamwise profile at y=W/4 !
     y_index = np.abs(yinterp[:, 0] - 0.25*W).argmin()
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-x-y=0.25W-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(xinterp[y_index,1:], columns=['x']).to_csv(csvname, index=False)
     csvname ='-R-x-y=0.25W-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(T_i[y_index,1:]/R, columns=['R']).to_csv(csvname, index=False)
     plt.plot(xinterp[y_index,1:]/W, T_i[y_index,1:]/R, c='black')
     plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
     plt.xlabel('$x/W$', fontsize=12)
     if inst==0:
        plt.ylabel('$\\overline{R}$/${R}_{0}$', fontsize=12)
     else:
        plt.ylabel('${R}$/${R}_{0}$', fontsize=12)
     filename='-surfaceplane-streamwise-salinity-profile-at-y=0.25W.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     print(f"#############################################################################################")
     print(f"Triangle mean velocities estimation based on 99%R defined triangle edge !")
     print(f"#############################################################################################")
     maskone = yinterp < 0 
     masktwo = np.isclose(T_i/R, 0.99, atol=1e-3)
     mask = maskone & masktwo
     trindexes = np.where(mask)
     trix = xinterp[trindexes]
     triy = yinterp[trindexes]
     Ux_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[0, Iplane]), (xinterp, yinterp), method='linear')
     Uy_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[1, Iplane]), (xinterp, yinterp), method='linear')
     ux=np.zeros((ngridy_surf,ngridx_surf))
     uy=np.zeros((ngridy_surf,ngridx_surf))
     for i in range (0,ngridy_surf):
       for j in range (0,ngridx_surf):
         c=Ux_i[i,j]
         if str(c) == '[nan]':
                ux[i,j] = 0
         else:
                ux[i,j] = c[0]
         d=Uy_i[i,j]
         if str(c) == '[nan]':
                uy[i,j] = 0
         else:
                uy[i,j] = d[0]

     triux = ux[trindexes]
     triuy = uy[trindexes]
     number = 5
     selection = np.linspace(0, len(triux) - 1, num=number, dtype=int)
     trix = trix.flatten()
     triux = triux.flatten()
     triuy = triuy.flatten()
     Ub=Reb*nu/H

     print(f"#######################################################################################################################")
     for i in selection:
         print(f'Ux values on the 99%R defined triangle edge for time={timename} at x={trix[i]} (m): Ux={triux[i]:.5f} (m/s)')
         print(f'Uy values on the 99%R defined triangle edge for time={timename} at x={trix[i]} (m): Uy={triuy[i]:.5f} (m/s)')
     print(f"#######################################################################################################################")

     #Triangle streamwise velocity Ux/Ub 
     fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-trix-99R-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(trix, columns=['x']).to_csv(csvname, index=False)
     csvname ='-triux-99R-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(triux, columns=['Ux']).to_csv(csvname, index=False)
     csvname ='-triux-99R-normalized-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(triux/Ub, columns=['Ux/Ub']).to_csv(csvname, index=False)
     ax.plot(trix/W, triux/Ub, c='black', label='$\\overline{U_x/U_b}$', markersize=5, linestyle='-')
     ax.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
     ax.set_xlabel('$x/W$', fontsize=12)
     if inst==0:
        ax.set_ylabel('$\\overline{U_x/U_b}$', fontsize=12)
     else: 
        ax.set_ylabel('${U_x/U_b}$', fontsize=12)
     filename = '-99R-triangle-streamwise-velocity.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully!")

     #Triangle spanwise velocity Uy/Ub 
     fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-triy-99R-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(trix, columns=['x']).to_csv(csvname, index=False)
     csvname ='-triux-99R-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(triuy, columns=['Ux']).to_csv(csvname, index=False)
     csvname ='-triuy-99R-normalized-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(triuy/Ub, columns=['Uy/Ub']).to_csv(csvname, index=False)
     ax.plot(trix/W, triuy/Ub, c='black', label='$\\overline{U_y/U_b}$', markersize=5, linestyle='-')
     ax.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
     ax.set_xlabel('$x/W$', fontsize=12)
     if inst==0:
        ax.set_ylabel('$\\overline{U_y/U_b}$', fontsize=12)
     else: 
        ax.set_ylabel('${U_y/U_b}$', fontsize=12)
     filename = '-99R-triangle-spanwise-velocity.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully!")

   #Surface sediment concentration map !
   if flagasurf == 1 and sedfoam==1:
     print(f"############################################################################################")
     print(f"Surface plane sediment concentration contourmap !")
     print(f"############################################################################################")
     print(f"Plunging point based on 99%a0 criterion !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     a_i = griddata((x[Iplane], y[Iplane]), alpha[Iplane], (xinterp, yinterp), method='linear')

     for m in range (0,len(a_i[ngridy_surf//2])):
       if a_i[ngridy_surf//2][m]<=0.99*a0:
          xp=xinterp[0][m]
          break
     print(f"###################################################################################################################")
     print(f'Plunging point position based on 99%a0 criterion for time={timename} x_p={xp:.5f} m')
     print(f'Plunging point position based on 99%a0 criterion for time={timename} normalized by the channel width x_p/W={(xp/W):.5f}')
     print(f"###################################################################################################################") 

     xvals = xinterp[int(ngridy_surf/2), 1:][::4]
     avals = a_i[int(ngridy_surf/2), 1:][::4]
     dadx  = np.gradient(avals, xvals)  
     mask = xvals > 0.5
     idx_min = np.argmin(dadx[mask])
     xsdrop = xvals[mask][idx_min]

     print(f"###################################################################################################################")
     print(f'Plunging point position based on the minimum of slope of the concentration profile criterion for time={timename} x_s={xsdrop:.5f} m')
     print(f'Plunging point position based on the minimum of slope of the concentration profile criterion for time={timename} normalized by the channel width x_s/W={(xsdrop/W):.5f}')
     print(f"###################################################################################################################") 

     #Surface sediment concentration field !
     fig = plt.figure(figsize=(6, 4),dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     plt.contourf(xinterp/W, yinterp/W, a_i/a0, cmap=scolormap, levels=np.linspace(-0.00000001, 1, 100))
     cbar = plt.colorbar()
     plt.tick_params(axis='x', labelsize=24)  
     plt.tick_params(axis='y', labelsize=24)
     ax=plt.gca()
     if inst==0:
        cbar.set_label('$\\overline{\\alpha}/a_0$', fontsize=28)
     else: 
        cbar.set_label('${\\alpha}/a_0$', fontsize=28)
     cbar.ax.tick_params(labelsize=24)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$y/W$', fontsize=28)
     filename='-surfaceplane-sediment-concentration-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Surface sediment concentration streamwise profile !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(xinterp[int(ngridy_surf/2),1:], columns=['x']).to_csv(csvname, index=False)
     csvname ='-alpha-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(a_i[int(ngridy_surf/2),1:], columns=['a']).to_csv(csvname, index=False)
     plt.plot(xinterp[int(ngridy_surf/2),1:]/W, a_i[int(ngridy_surf/2),1:], c='black')
     plt.xlabel('$x/W$', fontsize=12)
     if inst==0:
        plt.ylabel('$\\overline{\\alpha}$', fontsize=12)
     else: 
        plt.ylabel('${\\alpha}$', fontsize=12)
     plt.axhline(a0, ls='--', lw=2, c='k')
     plt.axvline(xsdrop/W, color='red', ls='--', lw=2)
     plt.xlim(0, xinterp[int(ngridy_surf/2),1:].max()/W)
     filename='-surfaceplane-streamwise-sediment-concentration-profile.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Surface sediment concentration slope streamwise profile !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'], 'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(xvals, columns=['x']).to_csv(csvname, index=False)
     csvname = '-dalphadx-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(dadx, columns=['dadx']).to_csv(csvname, index=False)
     plt.plot(xvals/W, dadx, c='black')
     plt.xlabel('$x/W$', fontsize=12)
     if inst == 0:
          plt.ylabel('$d\\overline{\\alpha}/dx$', fontsize=12)
     else:
          plt.ylabel('$d\\alpha/dx$', fontsize=12)
     plt.axvline(xsdrop/W, color='red', ls='--', lw=2)
     plt.xlim(0, xinterp[int(ngridy_surf/2),1:].max()/W)
     filename = '-surfaceplane-streamwise-sediment-concentration-slope-profile.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Surface restress map !
   if flagresurf == 1 and inst != 1:
     print(f"############################################################################################")
     print(f"Surface restress contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     Rxy_i = griddata((x[Iplane], y[Iplane]), Rxy[Iplane], (xinterp, yinterp), method='linear')
      
     rxy = np.zeros((ngridy_surf, ngridx_surf))
     for i in range(ngridy_surf):
       for j in range(ngridx_surf):
         c = Rxy_i[i,j]
         if np.isnan(c):
             rxy[i,j] = 0
         else:
             rxy[i,j] = c.item()

     #Surface restress contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(rxy))
     plt.contourf(xinterp/W, yinterp/W, rxy, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     cbar.set_label('$\\overline{R_{xy}} ({Pa})$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     filename='-surfaceplane-re-stress-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Surface vorticity map !
   if flagvortsurf == 1:
     print(f"############################################################################################")
     print(f"Surface vorticity contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     CURLx_i = griddata((x[Iplane], y[Iplane]), np.transpose(CURL[0, Iplane]), (xinterp, yinterp), method='linear')
     CURLy_i = griddata((x[Iplane], y[Iplane]), np.transpose(CURL[1, Iplane]), (xinterp, yinterp), method='linear')
     CURLz_i = griddata((x[Iplane], y[Iplane]), np.transpose(CURL[2, Iplane]), (xinterp, yinterp), method='linear')
      
     curlx = np.zeros((ngridy_surf, ngridx_surf))
     curly = np.zeros((ngridy_surf, ngridx_surf))
     curlz = np.zeros((ngridy_surf, ngridx_surf))
     for i in range(ngridy_surf):
       for j in range(ngridx_surf):
         c = CURLx_i[i,j]
         d = CURLy_i[i,j]
         e = CURLz_i[i,j]
         if np.isnan(c):
             curlx[i,j] = 0
         else:
             curlx[i,j] = c.item()
         if np.isnan(d):
             curly[i,j] = 0
         else:
             curly[i,j] = d.item()
         if np.isnan(e):
             curlz[i,j] = 0
         else:
             curlz[i,j] = e.item()

     #Surface vertical vorticity contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(curlz))
     plt.contourf(xinterp/W, yinterp/W, curlz, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if inst == 1:
        cbar.set_label('${\\omega_z} (1/{s}^{2})$', fontsize=12)
     else:
        cbar.set_label('$\\overline{\\omega_z} (1/{s}^{2})$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     filename='-surfaceplane-vertical-vorticity-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Surface Q-criterion map !
   if flagqcritsurf == 1:
     print(f"############################################################################################")
     print(f"Surface Q-criterion contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     Qcrit_i = griddata((x[Iplane], y[Iplane]), Qcrit[Iplane], (xinterp, yinterp), method='linear')

     #Surface Q-criterion contourmap !
     fig = plt.figure(figsize=(6, 4),dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(Qcrit_i))
     if flagssurf == 1 and sedfoam==0:
        T_i = griddata((x[Iplane], y[Iplane]), T[Iplane], (xinterp, yinterp), method='linear')
        plt.contour(xinterp/W, yinterp/W, T_i/R, levels=[0.99], colors='black', linewidths=2)
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1.5)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        plt.plot(xp/W, 0, marker='*', color='black', markersize=10)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xp/W])
        ax_top.set_xticklabels(['$x_p$'],fontsize=24,color='black')
     plt.contourf(xinterp/W, yinterp/W, Qcrit_i, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if inst == 1:
        cbar.set_label('$Q = \\frac{1}{2} \\left( ||\\Omega||^2 - ||S||^2 \\right)$', fontsize=12)
     else:
        cbar.set_label('$\\overline{Q} = \\frac{1}{2} \\left( ||\\overline{\\Omega}||^2 - ||\\overline{S}||^2 \\right)$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     filename='-surfaceplane-Qcriterion-field.pdf'
     savename = sim + filename
     plt.savefig(savename)
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Surface pressure map !
   if flagpsurf == 1 and inst != 0:
     print(f"############################################################################################")
     print(f"Surface pressure contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     p_i = griddata((x[Iplane], y[Iplane]), p[Iplane], (xinterp, yinterp), method='linear')

     #Surface pressure contourmap !
     fig = plt.figure(figsize=(6, 4),dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(p_i))
     plt.contourf(xinterp/W, yinterp/W, p_i, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     cbar.set_label('${p}$ (Pa)', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     filename='-surfaceplane-pressure-field.pdf'
     savename = sim + filename
     plt.savefig(savename)
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")


   #Surface particle pressure map !
   if flagpasurf == 1 and inst != 0 and sedfoam == 1:
     print(f"############################################################################################")
     print(f"Surface particle pressure contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     pa_i = griddata((x[Iplane], y[Iplane]), pa[Iplane], (xinterp, yinterp), method='linear')

     #Surface particle pressure contourmap !
     fig = plt.figure(figsize=(6, 4),dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(pa_i))
     plt.contourf(xinterp/W, yinterp/W, pa_i, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     cbar.set_label('${pa}$ (Pa)', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     filename='-surfaceplane-particle-pressure-field.pdf'
     savename = sim + filename
     plt.savefig(savename)
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Surfaceplane turbulent viscosity map !
   if flagnutsurf == 1 and inst != 0 and sedfoam == 0:
     print(f"############################################################################################")
     print(f"Surface turbulent viscosity contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     nut_i = griddata((x[Iplane], y[Iplane]), nut[Iplane], (xinterp, yinterp), method='linear')

     #Surface turbulent viscosity contourmap !
     fig = plt.figure(figsize=(6, 4),dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(nut_i))
     plt.contourf(xinterp/W, yinterp/W, nut_i, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     cbar.set_label('${\\nu_t}$ $(m^2/s)$', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     cbar.ax.tick_params(labelsize=24)
     plt.tick_params(axis='x', labelsize=24) 
     plt.tick_params(axis='y', labelsize=24)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$y/W$', fontsize=28)
     filename='-surfaceplane-turbulent-viscosity-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches='tight')
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Surface computed eddy viscosity map !
   if flagcompeddviscsurf == 1 and inst != 1 and sedfoam == 0:
     print(f"############################################################################################")
     print(f"Surface computed eddy viscosity contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     Umxy_i = griddata((x[Iplane], y[Iplane]), Umxy[Iplane], (xinterp, yinterp), method='linear')
     Rxy_i = griddata((x[Iplane], y[Iplane]), Rxy[Iplane], (xinterp, yinterp), method='linear')
      
     rxy = np.zeros((ngridy_surf, ngridx_surf))
     for i in range(ngridy_surf):
       for j in range(ngridx_surf):
         c = Rxy_i[i,j]
         if np.isnan(c):
             rxy[i,j] = 0
         else:
             rxy[i,j] = c.item()

     umxy = np.zeros((ngridy_surf, ngridx_surf))
     for i in range(ngridy_surf):
       for j in range(ngridx_surf):
         d = Umxy_i[i,j]
         if np.isnan(c):
             umxy[i,j] = 0
         else:
             umxy[i,j] = d.item()

     #Surface mean velocity gradient contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(umxy))
     plt.contourf(xinterp/W, yinterp/W, np.abs(umxy), cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     cbar.set_label(r'$|\overline{\mathrm{d}U/\mathrm{d}y}| 1/s$', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     filename='-surfaceplane-mean-velocity-gradient-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Surface reynolds stress contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(rxy))
     plt.contourf(xinterp/W, yinterp/W, np.abs(rxy), cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     cbar.set_label(r'$|-\overline{u^\prime v^\prime}| m^2/s^2$', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     filename='-surfaceplane-reynolds-stress-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Surface computed eddy viscosity contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     mask = (np.abs(umxy) > 1e-2) & (np.abs(rxy) > 1e-5)
     edvi = np.full_like(umxy, np.nan)
     edvi[mask] = 1 / (np.abs(umxy[mask]) / np.abs(rxy[mask]))
     plt.contourf(xinterp/W, yinterp/W, edvi, cmap=ecolormap, levels=np.linspace(0, 0.01, 100))
     cbar = plt.colorbar()
     cbar.set_label(r'$|\frac{-\overline{u^{\prime}v^{\prime}}}{\overline{\mathrm{d}U/\mathrm{d}y}}| m^2/s$', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     filename='-surfaceplane-computed-eddy-viscosity-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Surface velocity map !
   if flagusurf == 1:
     print(f"############################################################################################")
     print(f"Surface velocity contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     Ux_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[0, Iplane]), (xinterp, yinterp), method='linear')
     Uy_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[1, Iplane]), (xinterp, yinterp), method='linear')

     ux=np.zeros((ngridy_surf,ngridx_surf))
     uy=np.zeros((ngridy_surf,ngridx_surf))
     for i in range (0,ngridy_surf):
       for j in range (0,ngridx_surf):
         c=Ux_i[i,j]
         if str(c) == '[nan]':
                ux[i,j] = 0
         else:
                ux[i,j] = c[0]
         d=Uy_i[i,j]
         if str(d) == '[nan]':
                uy[i,j] = 0
         else:
                uy[i,j] = d[0]

     ux_avg=np.mean(ux) 
     uy_avg=np.mean(uy)
     Ub=Reb*nu/H

     print(f"#################################################")
     print(f'Ux(min) at the surface: %f ' % ux.min() , 'm/s')
     print(f'Ux(max) at the surface: %f ' % ux.max(), 'm/s')
     print(f"Ux average at the surface: {ux_avg:.4f} m/s")
     print(f'Uy(min) at the surface: %f ' % uy.min() , 'm/s')
     print(f'Uy(max) at the surface: %f ' % uy.max(), 'm/s')
     print(f"Uy average at the surface: {uy_avg:.4f} m/s")
     print(f"#################################################")

     #Streamwise surface velocity contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.max(np.abs(ux/Ub))
     plt.contourf(xinterp/W, yinterp/W, ux/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        T_i = griddata((x[Iplane], y[Iplane]), T[Iplane], (xinterp, yinterp), method='linear')
        plt.contour(xinterp/W, yinterp/W, T_i/R, levels=[0.99], colors='black', linewidths=2)
        plt.plot(xp / W, 0, marker='*', color='black', markersize=10)
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1.5)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xp/W])
        ax_top.set_xticklabels(['$x_p$'],fontsize=24,color='black')
     ycenterline = np.argmin(np.abs(yinterp[:,0])) 
     uxcenterline = ux[ycenterline, :] / Ub
     xcenterline = xinterp[ycenterline, :] / W
     sign_change_index = np.where((uxcenterline[:-1] > 0) & (uxcenterline[1:] < 0))[0]
     if sign_change_index.size > 0:
        xcrosssign = xcenterline[sign_change_index[0] + 1]
        plt.axvline(x=xcrosssign, color='black', linestyle='-.', linewidth=1.5)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xcrosssign])
        ax_top.set_xticklabels(['$x_w$'],fontsize=24,color='black')
     else:
        xcrosssign = 0
        plt.axvline(x=xcrosssign, color='black', linestyle='-.', linewidth=1.5)
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xcrosssign])
        ax_top.set_xticklabels(['$x_w$'],fontsize=24,color='black')
     if inst==0:
        cbar.set_label('$\\overline{u}/{U_0}$', fontsize=28)
     else:
        cbar.set_label('${u}/U_0$', fontsize=28)
     cbar.ax.tick_params(labelsize=24)
     cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
     quiver_spacing = 30
     q = plt.quiver(xinterp[0,::quiver_spacing]/W, yinterp[:,0][::quiver_spacing]/W, ux[::quiver_spacing,::quiver_spacing]/Ub, uy[::quiver_spacing,::quiver_spacing]/Ub, width=0.004, scale=25, headwidth=2, color='black')
     plt.tick_params(axis='x', labelsize=24)  
     plt.tick_params(axis='y', labelsize=24)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$y/W$', fontsize=28)
     filename='-surfaceplane-streamwise-velocity-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     print(f"######################################################################################################")
     print(f"Wake end point position based on ux=0 criterion for time={timename} x_w/W={xcrosssign}")
     print(f"#######################################################################################################")

     #Spanwise surface velocity contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.max(np.abs(uy/Ub))
     plt.contourf(xinterp/W, yinterp/W, uy/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xp/W])
        ax_top.set_xticklabels(['$x_p$'],fontsize=24,color='black')
        y_arrow = np.max(yinterp/W) * 0.9
        arrow_shift = 0.08  
        text_shift = 0.05   
        distance_steady_from_ymax = y_arrow - np.max(yinterp/W) * 0.1 - 1
        y_arrow_unsteady_new = -1 - distance_steady_from_ymax
        plt.arrow(xp/W - arrow_shift, y_arrow - np.max(yinterp/W) * 0.1, -0.04, 0, head_width=0.02, head_length=0.02, fc='white', ec='white')
        plt.plot([xp/W, xp/W - arrow_shift], [y_arrow - np.max(yinterp/W) * 0.1, y_arrow - np.max(yinterp/W) * 0.1], color='white', linewidth=1)
        plt.text(xp/W - text_shift, y_arrow, r'\textit{steady}', fontsize=14, ha='right', va='center', color='white')
        plt.arrow(xp/W + arrow_shift, y_arrow_unsteady_new - np.max(yinterp/W) * 0.1, 0.04, 0, head_width=0.02, head_length=0.02, fc='white', ec='white')
        plt.plot([xp/W, xp/W + arrow_shift], [y_arrow_unsteady_new - np.max(yinterp/W) * 0.1, y_arrow_unsteady_new - np.max(yinterp/W) * 0.1], color='white', linewidth=1)
        plt.text(xp/W + text_shift, y_arrow_unsteady_new, r'\textit{unsteady}', fontsize=14, ha='left', va='center', color='white')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if inst==0:
        cbar.set_label('$\\overline{v}/U_0$', fontsize=28)
     else:
        cbar.set_label('${v}/U_0$', fontsize=28)
     cbar.ax.tick_params(labelsize=24)
     cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
     formatter = ScalarFormatter(useMathText=True)
     plt.tick_params(axis='x', labelsize=24)  
     plt.tick_params(axis='y', labelsize=24)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$y/W$', fontsize=28)
     filename='-surfaceplane-spanwise-velocity-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Surface Ux streamwise profile at y=0 !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(xinterp[int(ngridy_surf/2),1:], columns=['x']).to_csv(csvname, index=False)
     csvname ='-Ux-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(ux[int(ngridy_surf/2),1:], columns=['Ux']).to_csv(csvname, index=False)
     plt.plot(xinterp[int(ngridy_surf/2),1:]/W, ux[int(ngridy_surf/2),1:], c='black')
     plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
     plt.xlabel('$x/W$', fontsize=12)
     if inst==0:
        plt.ylabel('$\\overline{U_x} (m/s)$', fontsize=12)
     else:
        plt.ylabel('${U_x} (m/s)$', fontsize=12)
     filename='-surfaceplane-streamwise-Ux-profile-at-y=0.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Surface Ux spanwise profile at x=0 !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-y-x=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(yinterp[1:,0], columns=['y']).to_csv(csvname, index=False)
     csvname ='-Ux-y-x=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(ux[1:,0], columns=['Ux']).to_csv(csvname, index=False)
     plt.plot(yinterp[1:,0]/W, ux[1:,0], c='black')
     plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     plt.xlabel('$y/W$', fontsize=12)
     if inst==0:
        plt.ylabel('$\\overline{U_x} (m/s)$', fontsize=12)
     else:
        plt.ylabel('${U_x} (m/s)$', fontsize=12)
     filename='-surfaceplane-spanwise-Ux-profile-at-x=0.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     if flagssurf == 1:
         #Surface Ux spanwise profile at x=0.5xp !
         fig = plt.figure(figsize=(6, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         csvname ='-y-x=0.5xp-z=0.csv'
         csvname = sim + csvname
         pd.DataFrame(yinterp[1:,0], columns=['y']).to_csv(csvname, index=False)
         csvname ='-Ux-y-x=0.5xp-z=0.csv'
         csvname = sim + csvname
         index = np.argmin(np.abs(xinterp - xp/2))
         pd.DataFrame(ux[1:,index], columns=['Ux']).to_csv(csvname, index=False)
         plt.plot(yinterp[1:,index]/W, ux[1:,index], c='black')
         plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
         plt.xlabel('$y/W$', fontsize=12)
         if inst==0:
            plt.ylabel('$\\overline{U_x} (m/s)$', fontsize=12)
         else:
            plt.ylabel('${U_x} (m/s)$', fontsize=12)
         filename='-surfaceplane-spanwise-Ux-profile-at-x=0.5xp.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Surface Ux spanwise profile at x=xp !
         fig = plt.figure(figsize=(6, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         csvname ='-y-x=xp-z=0.csv'
         csvname = sim + csvname
         pd.DataFrame(yinterp[1:,0], columns=['y']).to_csv(csvname, index=False)
         csvname ='-Ux-y-x=xp-z=0.csv'
         csvname = sim + csvname
         index = np.argmin(np.abs(xinterp - xp))
         pd.DataFrame(ux[1:,index], columns=['Ux']).to_csv(csvname, index=False)
         plt.plot(yinterp[1:,index]/W, ux[1:,index], c='black')
         plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
         plt.xlabel('$y/W$', fontsize=12)
         if inst==0:
            plt.ylabel('$\\overline{U_x} (m/s)$', fontsize=12)
         else:
            plt.ylabel('${U_x} (m/s)$', fontsize=12)
         filename='-surfaceplane-spanwise-Ux-profile-at-x=xp.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Surface Ux spanwise profile at x=xp+|xw-xp|/2 (midwake) !
         fig = plt.figure(figsize=(6, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         csvname ='-y-midwake-z=0.csv'
         csvname = sim + csvname
         pd.DataFrame(yinterp[1:,0], columns=['y']).to_csv(csvname, index=False)
         csvname ='-Ux-y-midwake-z=0.csv'
         csvname = sim + csvname
         index = np.argmin(np.abs(xinterp - (xp+(xcrosssign-xp)/2)))
         pd.DataFrame(ux[1:,index], columns=['Ux']).to_csv(csvname, index=False)
         plt.plot(yinterp[1:,index]/W, ux[1:,index], c='black')
         plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
         plt.xlabel('$y/W$', fontsize=12)
         if inst==0:
            plt.ylabel('$\\overline{U_x} (m/s)$', fontsize=12)
         else:
            plt.ylabel('${U_x} (m/s)$', fontsize=12)
         filename='-surfaceplane-spanwise-Ux-profile-at-midwake.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")


   #Centerplane velocity map !
   if flagucenter == 1:
      print(f"############################################################################################")
      print(f"Centerplane velocity contourmap !")
      print(f"############################################################################################")
      #Centerplane velocity contourmap !
      xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
      zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
      xinterp, zinterp = np.meshgrid(xi, zi)
      Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
      Ux_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[0, Iplane]), (xinterp, zinterp), method='linear')
      Uz_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[2, Iplane]), (xinterp, zinterp), method='linear')
      if sedfoam==0:
         T_i = griddata((x[Iplane], z[Iplane]), T[Iplane], (xinterp, zinterp), method='linear')
      else:
         a_i = griddata((x[Iplane], z[Iplane]), alpha[Iplane], (xinterp, zinterp), method='linear')

      ux=np.zeros((ngridz_center,ngridx_center))
      for i in range (0,ngridz_center):
          for j in range (0,ngridx_center):
             c=Ux_i[i,j]
             if str(c)=='[nan]':
                ux[i,j]=0
             else:
                ux[i,j]=c[0]

      uz=np.zeros((ngridz_center,ngridx_center))
      for i in range (0,ngridz_center):
          for j in range (0,ngridx_center):
              d=Uz_i[i,j]
              if str(d)=='[nan]':
                  uz[i,j]=0
              else:
                  uz[i,j]=d[0]

      for i in range (0,len(Ux_i)):
          for j in range (0,len(Ux_i[0])):
              if zinterp[i,j]<(-H-xinterp[i,j]*tan(beta/180*pi)):
                 ux[i,j]=np.nan
                 uz[i,j]=np.nan

      ux_avg=np.nanmean(ux) 
      uz_avg=np.nanmean(uz)
      Ub=Reb*nu/H
      Uc = np.nanmean(ux, axis=0)

      ut = (ux**2 + uz**2)**0.5
      epsilon = 1e-12 
      gamma = np.arctan(uz / (ux + epsilon)) 
      alphagon = beta/180*pi - gamma 
      us = np.cos(alphagon)*ut
      un = np.sin(alphagon)*ut
      us[ux < 0] = -np.abs(us[ux < 0])

      us[np.isnan(us) | np.isinf(us)] = 0
      un[np.isnan(un) | np.isinf(un)] = 0

      print(f"#################################################")
      print(f'Ux(min) at the centerplane: %f ' % np.nanmin(ux), 'm/s')
      print(f'Ux(max) at the centerplane: %f ' % np.nanmax(ux), 'm/s')
      print(f"Ux average at the centerplane: {np.nanmean(ux):.4f} m/s")
      print(f'Uz(min) at the centerplane: %f ' % np.nanmin(uz), 'm/s')
      print(f'Uz(max) at the centerplane: %f ' % np.nanmax(uz), 'm/s')
      print(f"Uz average at the centerplane: {np.nanmean(uz):.4f} m/s")
      print(f"#################################################")

      print(f"############################################################################################")
      print(f"hc(x) - plume height curve !")
      print(f"############################################################################################")
      dz_hc = np.diff(zinterp, axis=0)
      uxmdz = ux[1:] * dz_hc
      uxm2dz = (ux[1:] ** 2) * dz_hc
      int_uxmdz = np.nansum(uxmdz, axis=0)
      int_uxm2dy = np.nansum(uxm2dz, axis=0)
      epsilon = 1e-12

      hc = (int_uxmdz ** 2) / (int_uxm2dy+epsilon)

      indices = np.where((xi > W) & (xi < Lxb-W))[0]
      hc_in_range = hc[indices]
      if hc_in_range.size == 0:
         min_value = np.nan
         min_index = None
      else:
         min_value = np.min(hc_in_range)
         min_index = indices[np.argmin(hc_in_range)]
      indices = np.where((xi > W) & (xi < xi[min_index]))[0]
      hc_in_range = hc[indices]
      if hc_in_range.size == 0:
         max_value = np.nan
         max_index = None
      else:
         max_value = np.max(hc_in_range)
         max_index = indices[np.argmax(hc_in_range)]

      if sedfoam == 0:
         xd = xi[max_index]
         xumin = xi[min_index]
      elif sedfoam == 1:
         xd = 0
         xumin = 0

      #hc(x) curve !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      csvname ='-x-coordinate-hc.csv'
      csvname = sim + csvname
      pd.DataFrame(xi[1:ngridx_center-2], columns=['x']).to_csv(csvname, index=False)
      csvname ='-hc-coordinate-hc.csv'
      csvname = sim + csvname
      pd.DataFrame(hc[1:ngridx_center-2], columns=['hc']).to_csv(csvname, index=False)          
      plt.plot(xi[1:ngridx_center-2]/W, hc[1:ngridx_center-2], '-', linewidth=2, color='black', label=f'{sim}')
      plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
      plt.axvline(x=xumin/W, color='black', linestyle=':', linewidth=1)
      plt.xlabel('$x/W$', fontsize=12)
      plt.ylabel('$h_c (m)$', fontsize=12)
      filename='-curve-plume-height.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      print(f"############################################################################################")
      print(f"xuc undercurrent point !")
      print(f"############################################################################################")

      if sedfoam == 0:
          for i in range (0,len(T_i)):
            for j in range (0,len(T_i[0])):
               if zinterp[i,j]<(-H-xinterp[i,j]*tan(beta/180*pi)):
                  T_i[i,j]=np.nan
      
          xs=np.array([0, xinterpmax_center])
          ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
          cf = plt.contour(xinterp, zinterp, T_i / R, levels=[0.99], colors='red', linewidths=2)
          paths = cf.get_paths()
          xvals = []
          def distance_to_line(x, y, x1, y1, x2, y2):
            numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            return numerator / denominator
          for path in paths:
           vertices = path.vertices 
           for vertex in vertices:
            xv, zv = vertex
            dist = distance_to_line(xv, zv, xs[0], ys[0], xs[1], ys[1])
            if dist > 1e-2: 
                xvals.append(xv)
          xuc = np.max(xvals)
      else:
          for i in range (0,len(a_i)):
            for j in range (0,len(a_i[0])):
               if zinterp[i,j]<(-H-xinterp[i,j]*tan(beta/180*pi)):
                  a_i[i,j]=np.nan
      
          xs=np.array([0, xinterpmax_center])
          ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
          cf = plt.contour(xinterp, zinterp, a_i / a0, levels=[0.99], colors='red', linewidths=2)
          paths = cf.get_paths()
          xvals = []
          def distance_to_line(x, y, x1, y1, x2, y2):
            numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            return numerator / denominator
          for path in paths:
           vertices = path.vertices 
           for vertex in vertices:
            xv, zv = vertex
            dist = distance_to_line(xv, zv, xs[0], ys[0], xs[1], ys[1])
            if dist > 1e-2: 
                xvals.append(xv)
          if len(xvals) == 0:
             xuc = 0
          else:
             xuc = np.max(xvals)


      if sedfoam == 0:
         print(f"############################################################################################")
         print(f"Interface of gravity current iso-contour !")
         print(f"############################################################################################")
         xcrossings = []
         zcrossings = []

         for j in range(0, len(ux[0]), 4):  
            for i in range(1, len(ux[:, j])):
               if ux[i-1, j] > 0 and ux[i, j] < 0:
                  zcross = zinterp[i-1, j] + (ux[i-1, j] / (ux[i-1, j] - ux[i, j])) * (zinterp[i, j] - zinterp[i-1, j])
                  xcrossings.append(xinterp[i, j])
                  zcrossings.append(zcross)
                  break

         def zreference(x):
             return -H - x * np.tan(beta * pi / 180)

         verticaldistances = []
         for j in range(len(xcrossings)):
            xcross = xcrossings[j]
            zcross = zcrossings[j]
            zref = zreference(xcross)
            verticaldistance = abs(zcross - zref)
            verticaldistances.append(verticaldistance)
         verticaldistances = np.array(verticaldistances)
         Hg = verticaldistances * np.sin(90/180*pi-beta/180*pi)
         horizontaldistances = verticaldistances * np.sin(beta/180*pi)
         xshifted = xcrossings - horizontaldistances
         exclper = 0.25
         numvalexcl = int(len(Hg) * exclper)
         Hgmid = Hg[numvalexcl: -numvalexcl]
         Hgm = np.mean(Hgmid)
       
         euclidiandistances = []
         xshiftedvalues = []
         Ug = []
         Hgreduced = []
         Rg = []
         points = np.column_stack((xinterp.flatten(), zinterp.flatten()))
         usvalues = us.flatten()
         Rvalues = T_i.flatten()
         for j in range(0, len(xshifted), 5):  
           xslope = xshifted[j]
           zslope = zreference(xshifted[j])
           xsurf = xslope + abs(zslope) * np.tan(beta / 180 * pi)
           zsurf = 0
           xint = xcrossings[j]
           zint = zcrossings[j]
           num_points = 5  
           xnet = np.linspace(xint, xslope, num_points + 2)[1:-1]  
           znet = np.linspace(zint, zslope, num_points + 2)[1:-1]
           euclidiandistance = np.sqrt((xslope - xint) ** 2 + (zslope - zint) ** 2)
           euclidiandistances.append(euclidiandistance)
           xshiftedvalues.append(xshifted[j])
           usnet = griddata(points, usvalues, (xnet, znet), method='linear')
           usnetmean = np.mean(usnet)
           Ug.append(usnetmean)
           Hgreduced.append(Hg[j])
           Rgnet = griddata(points, Rvalues, (xnet, znet), method='linear')
           Rgnetmean = np.mean(Rgnet)
           Rg.append(Rgnetmean)
         exclper = 0.25
         numvalexcl = int(len(Ug) * exclper)
         Ugm = np.mean(Ug[numvalexcl: -numvalexcl])
         Hgreduced = np.array(Hgreduced, dtype=float)
         Frh = Ug/np.sqrt(Hgreduced*g)
         Frhm = Ugm/np.sqrt(Hgm*g)
         numvalexcl = int(len(Rg) * exclper)
         Rgm= np.mean(Rg[numvalexcl: -numvalexcl])
         Rg = np.array(Rg, dtype=float)
         Frdh = Ug/np.sqrt(Hgreduced*g*Rg)
         Frdhm=Frhm/Rgm**0.5

         #Single point estimation of Frdh at Hgmin location
         Hgmin = np.min(Hg)
         index_min = np.argmin(Hg)
         x_target = xshifted[index_min]
         xslope = x_target
         zslope = zreference(x_target)
         xint = xcrossings[index_min]
         zint = zcrossings[index_min]
         num_points = 5
         xnet = np.linspace(xint, xslope, num_points + 2)[1:-1]
         znet = np.linspace(zint, zslope, num_points + 2)[1:-1]
         points = np.column_stack((xinterp.flatten(), zinterp.flatten()))
         usvalues = us.flatten()
         Rvalues = T_i.flatten()
         usnet = griddata(points, usvalues, (xnet, znet), method='linear')
         Ugsingle = np.mean(usnet)
         Rgnet = griddata(points, Rvalues, (xnet, znet), method='linear')
         Rgsingle = np.mean(Rgnet)
         Frhsingle = Ugsingle / np.sqrt(g * Hgmin)
         Frdhsingle = Ugsingle / np.sqrt(g * Hgmin * Rgsingle)
               
         #Hg(x) curve !
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         csvname ='-x-coordinate-Hg.csv'
         csvname = sim + csvname
         pd.DataFrame(xshifted, columns=['x']).to_csv(csvname, index=False)
         csvname ='-Hg-coordinate-Hg.csv'
         csvname = sim + csvname
         pd.DataFrame(Hg, columns=['Hg']).to_csv(csvname, index=False)          
         plt.plot(xshifted/W, Hg, '-', linewidth=2, color='black', label=f'{sim}')
         plt.scatter(np.array(xshiftedvalues)/W, euclidiandistances, c='red', s=80)
         #plt.plot(xcrossings, Hg, '-', linewidth=2, color='red', label=f'{sim}')
         plt.axhline(y=Hgm, color='black', linestyle='--')
         plt.xlabel('$x/W$', fontsize=12)
         plt.ylabel('$H_g (m)$', fontsize=12)
         plt.ylim(0, 1)
         filename='-curve-gravity-current-height.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Ug(x) curve !
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         csvname ='-x-coordinate-Ug.csv'
         csvname = sim + csvname
         pd.DataFrame(xshiftedvalues, columns=['x']).to_csv(csvname, index=False)
         csvname ='-Ug-coordinate-Ug.csv'
         csvname = sim + csvname
         pd.DataFrame(Ug, columns=['Ug']).to_csv(csvname, index=False)          
         plt.plot(np.array(xshiftedvalues)/W, Ug, '-', linewidth=2, color='black', label=f'{sim}')
         plt.axhline(y=Ugm, color='black', linestyle='--')
         plt.xlabel('$x/W$', fontsize=12)
         plt.ylabel('$U_g (m/s)$', fontsize=12)
         filename='-curve-gravity-current-mean-velocity.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Rg(x) curve !
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         csvname ='-x-coordinate-Ug.csv'
         csvname = sim + csvname
         pd.DataFrame(xshiftedvalues, columns=['x']).to_csv(csvname, index=False)
         csvname ='-Rg-coordinate-Ug.csv'
         csvname = sim + csvname
         pd.DataFrame(Rg, columns=['Rg']).to_csv(csvname, index=False)          
         plt.plot(np.array(xshiftedvalues)/W, Rg/R*100, '-', linewidth=2, color='black', label=f'{sim}')
         plt.axhline(y=Rgm/R*100, color='black', linestyle='--')
         plt.xlabel('$x/W$', fontsize=12)
         plt.ylabel('$R_g/R_0 \\%$', fontsize=12)
         filename='-curve-gravity-current-mean-salinity.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Frh(x) curve !
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         csvname ='-x-coordinate-Frh.csv'
         csvname = sim + csvname
         pd.DataFrame(xshiftedvalues, columns=['x']).to_csv(csvname, index=False)
         csvname ='-Frh-coordinate-Frh.csv'
         csvname = sim + csvname
         pd.DataFrame(Frh, columns=['Frh']).to_csv(csvname, index=False)          
         plt.plot(np.array(xshiftedvalues)/W, Frh, '-', linewidth=2, color='black', label=f'{sim}')
         plt.axhline(y=Frhm, color='black', linestyle='--')
         plt.xlabel('$x/W$', fontsize=12)
         plt.ylabel('$Fr_h$', fontsize=12)
         filename='-curve-gravity-current-hydraulic-Froude-number.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Frdh(x) curve !
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         csvname ='-x-coordinate-Frdh.csv'
         csvname = sim + csvname
         pd.DataFrame(xshiftedvalues, columns=['x']).to_csv(csvname, index=False)
         csvname ='-Frdh-coordinate-Frdh.csv'
         csvname = sim + csvname
         pd.DataFrame(Frdh, columns=['Frdh']).to_csv(csvname, index=False)          
         plt.plot(np.array(xshiftedvalues)/W, Frdh, '-', linewidth=2, color='black', label=f'{sim}')
         plt.axhline(y=Frdhm, color='black', linestyle='--')
         plt.xlabel('$x/W$', fontsize=12)
         plt.ylabel('$Frd_h$', fontsize=12)
         filename='-curve-gravity-current-hydraulic-densimetric-Froude-number.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         print(f"######################################################################################################")
         print(f'Global max value of hc: {np.nanmax(hc):.4f} m')
         print(f'Global min value of hc: {np.nanmin(hc):.4f} m')
         print(f"Local max value of hc: {max_value:.5f} m")
         print(f"Local min value of hc: {min_value:.5f} m")
         print(f"Plunging point position based on hc(x) maximum criterion for time={timename} x_d={xd} m")
         print(f"Minimum hc(x) position for time={timename} x_ud={xumin} m")
         print(f"Plunging point position based on hc(x) maximum criterion normalized by the channel width x_d/W={xd/W} m")
         print(f"Minimum hc(x) position normalized by the channel width x_ud/W={xumin/W} m")
         print(f"Undercurrent start location x_uc={xuc} m") 
         print(f"Mean height of the gravity current H_g={Hgm} m") 
         print(f"Mean velocity of the gravity current U_g={Ugm} m/s") 
         print(f"Mean hydraulic Froude number of the gravity current Fr_h={Frhm}")
         print(f"Mean realtive density difference of the gravity current Rgm/R0={Rgm/R*100} %")  
         print(f"Mean densimetric hydraulic Froude number of the gravity current Frd_h={Frdhm}")  
         print(f"Minimum height of the gravity current H_g={Hgmin} m") 
         print(f"Mean velocity of the gravity current at the minimum height location U_g={Ugsingle} m/s") 
         print(f"Mean hydraulic Froude number of the gravity current at the minimum height location Fr_h={Frhsingle}")
         print(f"Mean realtive density difference of the gravity current at the minimum height location Rgm/R0={Rgsingle/R*100} %")  
         print(f"Mean densimetric hydraulic Froude number of the gravity current at the minimum height location Frd_h={Frdhsingle}") 
         print(f"#######################################################################################################")

      #Streamwise centerplane velocity contourmap !
      fig = plt.figure(figsize=(16, 2), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'          
      xs=np.array([0, xinterpmax_center])
      ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
      plt.plot(xs/W,ys/H,c='black')
      abs_max = np.nanmax(np.abs(ux))
      contour = plt.contourf(xinterp/W, zinterp/H, ux, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
      if sedfoam == 0:
         plt.plot(np.array(xcrossings)/W, np.array(zcrossings)/H, color='black', linewidth=2)
         for j in range(0, len(xshifted), 5):  
           xslope = xshifted[j]
           zslope = zreference(xshifted[j])
           xsurf = xslope + abs(zslope) * np.tan(beta / 180 * pi)
           zsurf = 0
           xint = xcrossings[j]
           zint = zcrossings[j]
           plt.plot([xint/W, xslope/W], [zint/H, zslope/H], 'k--')
           if j == index_min:
              plt.plot(xint/W, zint/H, 'go', markersize=10, markeredgecolor='k', linewidth=2)  
           else:
              plt.plot(xint/W, zint/H, 'ro', markersize=8)
           num_points = 5  
           xnet = np.linspace(xint, xslope, num_points + 2)[1:-1]  
           znet = np.linspace(zint, zslope, num_points + 2)[1:-1]
           plt.plot(xnet/W, znet/H, 'bo', markersize=5) 
      if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
      if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
      slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
      y_at_xd = slope * (xd - xs[0]) + ys[0]
      plt.plot([xd/W, xd/W], [y_at_xd/H, 0], color='black', linestyle=':', linewidth=1)
      cbar = plt.colorbar(contour)
      if inst==0:
         cbar.set_label('$\\overline{U_x} (m/s)$', fontsize=28)
      else:
         cbar.set_label('${U_x} (m/s)$', fontsize=28)
      cbar.ax.tick_params(labelsize=10)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.xlabel('$x/W$', fontsize=28)
      plt.ylabel('$z/H_0$', fontsize=28)
      plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
      filename='-centerplane-streamwise-velocity-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Slopewise centerplane velocity contourmap !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'          
      xs=np.array([0, xinterpmax_center])
      ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
      plt.plot(xs/W,ys/H,c='black') 
      abs_max = np.nanmax(np.abs(ux))
      contour = plt.contourf(xinterp/W, zinterp/H, us, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
      quiver_spacing = 30
      q = plt.quiver(xinterp[0,::quiver_spacing]/W, zinterp[:,0][::quiver_spacing//4]/H, ux[::quiver_spacing//4,::quiver_spacing], uz[::quiver_spacing//4,::quiver_spacing], width=0.002, scale=1.5, headwidth=2, color='black')
      if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
      if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
      slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
      y_at_xd = slope * (xd - xs[0]) + ys[0]
      plt.plot([xd/W, xd/W], [y_at_xd/H, 0], color='black', linestyle=':', linewidth=1)
      cbar = plt.colorbar(contour)
      if inst==0:
         cbar.set_label('$\\overline{U_s} (m/s)$', fontsize=28)
      else:
         cbar.set_label('${U_s} (m/s)$', fontsize=28)
      cbar.ax.tick_params(labelsize=10)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.xlabel('$x/W$', fontsize=28)
      plt.ylabel('$z/H_0$', fontsize=28)
      plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
      filename='-centerplane-slopewise-velocity-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Vertical centerplane velocity contourmap !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'], 'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'           
      xs = np.array([0, xinterpmax_center])
      ys = np.array([-H, -H - xinterpmax_center * tan(beta / 180 * pi)])
      plt.plot(xs / W, ys / H, c='black')
      abs_max = 0.5
      plt.contourf(xinterp / W, zinterp / H, uz / Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
      cbar = plt.colorbar()
      def distance_to_line(x, y, x1, y1, x2, y2):
            numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            return numerator / denominator
      if sedfoam==0:
          Tfilt = T_i 
          for i in range(ngridz_center):
              for j in range(ngridx_center):
                    dist = distance_to_line(xinterp[i, j], zinterp[i, j], xs[0], ys[0], xs[1], ys[1])
                    if dist < 1e-2 or zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                        Tfilt[i, j] = 0 
          plt.contour(xinterp / W, zinterp / H, Tfilt / R, levels=[0.99], colors='red', linewidths=2)
          plt.contour(xinterp / W, zinterp / H, T_i / R, levels=[0.01], colors='black', linewidths=2)
      else: 
          afilt = a_i 
          for i in range(ngridz_center):
              for j in range(ngridx_center):
                    dist = distance_to_line(xinterp[i, j], zinterp[i, j], xs[0], ys[0], xs[1], ys[1])
                    if dist < 1e-2 or zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                        afilt[i, j] = 0 
          plt.contour(xinterp / W, zinterp / H, afilt, levels=[0.99 * a0], colors='red', linewidths=2)
          plt.contour(xinterp / W, zinterp / H, afilt, levels=[0.01 * a0], colors='black', linewidths=2)
      slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
      y_at_xp = slope * (xuc - xs[0]) + ys[0]
      if flagssurf == 1 and sedfoam == 0:
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            y_at_xp = slope * (xp - xs[0]) + ys[0]
            plt.plot([xp / W, xp / W], [y_at_xp / H, 0], color='red', linestyle='--', linewidth=2.0)
            ax = plt.gca()
            ax_top = ax.secondary_xaxis('top')
            ax_top.set_xticks([xp / W])
            ax_top.set_xticklabels(['$x_p$'], fontsize=24, color='black')
      if flagasurf == 1 and sedfoam == 1:
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            y_at_xp = slope * (xp - xs[0]) + ys[0]
            plt.plot([xp / W, xp / W], [y_at_xp / H, 0], color='red', linestyle='--', linewidth=2.0)
      if flagusurf == 1:
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            y_at_xcrosssign = slope * (xcrosssign * W - xs[0]) + ys[0]
            plt.plot([xcrosssign, xcrosssign], [y_at_xcrosssign / H, 0], color='red', linestyle='-.', linewidth=2.0)
            ax = plt.gca()
            ax_top = ax.secondary_xaxis('top')
            ax_top.set_xticks([xcrosssign])
            ax_top.set_xticklabels(['$x_w$'], fontsize=24, color='black')
      if inst == 0:
            cbar.set_label('$\\overline{w}/U_0$', fontsize=28)
      else:
            cbar.set_label('${w}/U_0$', fontsize=28)
      cbar.ax.tick_params(labelsize=24)
      quiver_spacing = 30
      q = plt.quiver(xinterp[0, ::quiver_spacing] / W, zinterp[:, 0][::quiver_spacing // 4] / H, ux[::quiver_spacing // 4, ::quiver_spacing] / Ub, uz[::quiver_spacing // 4, ::quiver_spacing] / Ub, width=0.002, scale=25, headwidth=2, color='black')
      plt.tick_params(axis='x', labelsize=24) 
      plt.tick_params(axis='y', labelsize=24)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
      cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      ax = plt.gca()
      if flagssurf == 1 and flagusurf ==1 and sedfoam == 0:
         ax_top = ax.secondary_xaxis('top')
         ax_top.set_xticks([xp / W, xcrosssign])
         ax_top.set_xticklabels(['$x_p$', '$x_w$'], fontsize=24, color='black')
         ax_top.set_xlim(0, xinterpmax_center / W)
      ax.tick_params(axis='x', labelsize=24)
      plt.xlabel('$x/W$', fontsize=28)
      plt.ylabel('$z/H_0$', fontsize=28)
      plt.ylim((-H - xinterpmax_center * tan(beta / 180 * pi)) / H, 0)
      filename = '-centerplane-vertical-velocity-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Centerplane depth-averaged streamwise velocity profile Uc as function of x !
      fig = plt.figure(figsize=(6, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      csvname ='-x-y=0.csv'
      csvname = sim + csvname
      pd.DataFrame(xinterp[0,:]/W, columns=['x']).to_csv(csvname, index=False)
      csvname ='-Uc-x-y=0.csv'
      csvname = sim + csvname
      pd.DataFrame(Uc[:], columns=['Uc']).to_csv(csvname, index=False)
      plt.plot(xinterp[0,:]/W, Uc[:], c='black')
      plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
      plt.xlabel('$x/W$', fontsize=12)
      if inst==0:
         plt.ylabel('$\\overline{U_c} (m/s)$', fontsize=12)
      else: 
         plt.ylabel('${U_c} (m/s)$', fontsize=12)
      filename='-depth-averaged-streamwise-velocity-profile-as-function-of-x.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Centerplane Ux vertical profile at x=0 !
      fig = plt.figure(figsize=(6, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      csvname ='-z-x=0-y=0.csv'
      csvname = sim + csvname
      pd.DataFrame(zinterp[:-1,0]/H, columns=['z']).to_csv(csvname, index=False)
      csvname ='-Ux-z-x=0-y=0.csv'
      csvname = sim + csvname
      pd.DataFrame(ux[:-1,0], columns=['Ux']).to_csv(csvname, index=False)
      plt.plot(ux[:-1,0], zinterp[:-1,0]/H, c='black')
      plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
      plt.ylabel('$z/H_0$', fontsize=12)
      if inst==0:
         plt.xlabel('$\\overline{U_x} (m/s)$', fontsize=12)
      else: 
         plt.xlabel('${U_x} (m/s)$', fontsize=12)
      filename='-centerplane-vertical-Ux-profile-at-x=0.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      if flagssurf == 1 and sedfoam==0:
          #Centerplane Ux vertical profile at x=xp/2 !
          fig = plt.figure(figsize=(6, 4), dpi=500)
          plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
          plt.rcParams['xtick.direction'] = 'in'
          plt.rcParams['ytick.direction'] = 'in'
          csvname ='-z-x=0.5xp-y=0.csv'
          csvname = sim + csvname
          index = np.argmin(np.abs(xinterp - xp/2))
          pd.DataFrame(zinterp[:-1,index], columns=['z']).to_csv(csvname, index=False)
          csvname ='-Ux-z-0.5xp-y=0.csv'
          csvname = sim + csvname
          pd.DataFrame(ux[:-1,index], columns=['Ux']).to_csv(csvname, index=False)
          plt.plot(ux[:-1,index], zinterp[:-1,index]/H, c='black')
          plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
          plt.ylabel('$z/H_0$', fontsize=12)
          if inst==0:
             plt.xlabel('$\\overline{U_x} (m/s)$', fontsize=12)
          else:
             plt.xlabel('${U_x} (m/s)$', fontsize=12)
          filename='-centerplane-vertical-Ux-profile-at-x=0.5xp.pdf'
          savename = sim + filename
          plt.savefig(savename, bbox_inches="tight")
          plt.close()
          print(f"Plotted \"{savename}\" successfully !")
      
          #Centerplane Ux vertical profile at x=xp !
          fig = plt.figure(figsize=(6, 4), dpi=500)
          plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
          plt.rcParams['xtick.direction'] = 'in'
          plt.rcParams['ytick.direction'] = 'in'
          csvname ='-z-x=xp-y=0.csv'
          csvname = sim + csvname
          index = np.argmin(np.abs(xinterp - xp))
          pd.DataFrame(zinterp[:-1,index], columns=['z']).to_csv(csvname, index=False)
          csvname ='-Ux-z-x=xp-y=0.csv'
          csvname = sim + csvname
          pd.DataFrame(ux[:-1,index], columns=['Ux']).to_csv(csvname, index=False)
          plt.plot(ux[:-1,index], zinterp[:-1,index]/H, c='black')
          plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
          plt.ylabel('$z/H_0$', fontsize=12)
          if inst==0:
             plt.xlabel('$\\overline{U_x} (m/s)$', fontsize=12)
          else:
             plt.xlabel('${U_x} (m/s)$', fontsize=12)
          filename='-centerplane-vertical-Ux-profile-at-x=xp.pdf'
          savename = sim + filename
          plt.savefig(savename, bbox_inches="tight")
          plt.close()
          print(f"Plotted \"{savename}\" successfully !")

   #Centerplane streamlines map !
   if flagstreamcenter == 1:
      print(f"############################################################################################")
      print(f"Centerplane streamlines contourmap !")
      print(f"############################################################################################")
      #Centerplane streamlines contourmap !
      xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
      zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
      xinterp, zinterp = np.meshgrid(xi, zi)
      Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
      Ux_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[0, Iplane]), (xinterp, zinterp), method='linear')
      Uz_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[2, Iplane]), (xinterp, zinterp), method='linear')

      ux=np.zeros((ngridz_center,ngridx_center))
      for i in range (0,ngridz_center):
          for j in range (0,ngridx_center):
             c=Ux_i[i,j]
             if str(c)=='[nan]':
                ux[i,j]=0
             else:
                ux[i,j]=c[0]

      uz=np.zeros((ngridz_center,ngridx_center))
      for i in range (0,ngridz_center):
          for j in range (0,ngridx_center):
              d=Uz_i[i,j]
              if str(d)=='[nan]':
                  uz[i,j]=0
              else:
                  uz[i,j]=d[0]

      for i in range (0,len(Ux_i)):
          for j in range (0,len(Ux_i[0])):
              if zinterp[i,j]<(-H-xinterp[i,j]*tan(beta/180*pi)):
                 ux[i,j]=np.nan
                 uz[i,j]=np.nan

      ux_avg=np.nanmean(ux) 
      uz_avg=np.nanmean(uz)

      #Centerplane streamlines velocity contourmap !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'          
      xs=np.array([0, xinterpmax_center])
      ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
      plt.plot(xs/W,ys/H,c='black') 
      abs_max = np.nanmax(np.abs(ux))
      contour = plt.contourf(xinterp/W, zinterp/H, ux, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
      speed = np.sqrt(np.nan_to_num(ux, nan=0.0, posinf=0.0, neginf=0.0)**2 + np.nan_to_num(uz, nan=0.0, posinf=0.0, neginf=0.0)**2)
      plt.streamplot(xinterp/W, zinterp/H, ux, uz, color='black', linewidth=2*speed/speed.max())
      cbar = plt.colorbar(contour)
      if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
      if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
      if flagucenter == 1:
         slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
         y_at_xd = slope * (xd - xs[0]) + ys[0]
         plt.plot([xd/W, xd/W], [y_at_xd/H, 0], color='black', linestyle=':', linewidth=1)
      if inst==0:
         cbar.set_label('$\\overline{U_x} (m/s)$', fontsize=28)
      else:
         cbar.set_label('${U_x} (m/s)$', fontsize=28)
      cbar.ax.tick_params(labelsize=10)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.xlabel('$x/W$', fontsize=28)
      plt.ylabel('$z/H_0$', fontsize=28)
      plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
      filename='-centerplane-streamlines-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

   #Centerplane turbulent viscosity map !
   if flagnutcenter == 1 and inst != 0 and sedfoam == 0:
     print(f"############################################################################################")
     print(f"Centerplane turbulent viscosity contourmap !")
     print(f"############################################################################################")
     #Centerplane turbulent viscosity contourmap !
     xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
     zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
     xinterp, zinterp = np.meshgrid(xi, zi)
     Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
     nut_i = griddata((x[Iplane], z[Iplane]), nut[Iplane], (xinterp, zinterp), method='linear')

     #Centerplane turbulent viscosity contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     print("NaN values in p:", np.isnan(nut_i).any())
     print("Inf values in p:", np.isinf(nut_i).any())
     abs_max = np.nanmax(np.abs(nut_i))
     for i in range(ngridz_center):
         for j in range(ngridx_center):
             if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                nut_i[i, j] = 0 
     plt.contourf(xinterp/W, zinterp/H, nut_i, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     cbar.set_label('$\\nu_t$ ($m^2/s$)', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$z/H_0$', fontsize=28)
     plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
     filename='-centerplane-turbulent-viscosity-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !") 

   #Centerplane computed eddy viscosity map !
   if flagcompeddvisccenter == 1 and inst != 1 and sedfoam == 0:
     print(f"############################################################################################")
     print(f"Centerplane computed eddy viscosity contourmap !")
     print(f"############################################################################################")
     #Centerplane turbulent viscosity contourmap !
     xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
     zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
     xinterp, zinterp = np.meshgrid(xi, zi)
     Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
     Umxz_i = griddata((x[Iplane], z[Iplane]), Umxz[Iplane], (xinterp, zinterp), method='linear')
     Umxy_i = griddata((x[Iplane], z[Iplane]), Umxy[Iplane], (xinterp, zinterp), method='linear')
     Rxz_i = griddata((x[Iplane], z[Iplane]), Rxz[Iplane], (xinterp, zinterp), method='linear')
     Rxy_i = griddata((x[Iplane], z[Iplane]), Rxy[Iplane], (xinterp, zinterp), method='linear')

     rxz = np.zeros((ngridz_center, ngridx_center))
     for i in range(ngridz_center):
       for j in range(ngridx_center):
         c = Rxz_i[i,j]
         if np.isnan(c):
             rxz[i,j] = 0
         else:
             rxz[i,j] = c.item()

     umxz = np.zeros((ngridz_center, ngridx_center))
     for i in range(ngridz_center):
       for j in range(ngridx_center):
         c = Umxz_i[i,j]
         if np.isnan(c):
             umxz[i,j] = 0
         else:
             umxz[i,j] = c.item()

     rxy = np.zeros((ngridz_center, ngridx_center))
     for i in range(ngridz_center):
       for j in range(ngridx_center):
         c = Rxy_i[i,j]
         if np.isnan(c):
             rxy[i,j] = 0
         else:
             rxy[i,j] = c.item()

     umxy = np.zeros((ngridz_center, ngridx_center))
     for i in range(ngridz_center):
       for j in range(ngridx_center):
         c = Umxy_i[i,j]
         if np.isnan(c):
             umxy[i,j] = 0
         else:
             umxy[i,j] = c.item()

     #Centerplane mean velocity gradient contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     abs_max = np.nanmax(np.abs(umxz))
     for i in range(ngridz_center):
         for j in range(ngridx_center):
             if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                umxz[i, j] = 0 
     plt.contourf(xinterp/W, zinterp/H, np.abs(umxz), cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     cbar.set_label(r'$|\overline{\mathrm{d}U/\mathrm{d}z}| 1/s$', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$z/H_0$', fontsize=28)
     plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
     filename='-centerplane-mean-velocity-gradient-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !") 

     #Centerplane reynolds stress contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     abs_max = np.nanmax(np.abs(rxz))
     plt.contourf(xinterp/W, zinterp/H, np.abs(rxz), cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     cbar.set_label(r'$|-\overline{u^\prime w^\prime}| m^2/s^2$', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$z/H_0$', fontsize=28)
     plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
     filename='-centerplane-reynolds-stress-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !") 

     #Centerplane computed eddy viscosity contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     mask = (np.abs(umxy) > 1e-2) & (np.abs(rxy) > 1e-5)
     edvi = np.full_like(umxy, np.nan)
     edvi[mask] = 1 / (np.abs(umxy[mask]) / np.abs(rxy[mask]))
     plt.contourf(xinterp/W, zinterp/H, edvi, cmap=ecolormap, levels=np.linspace(0, 0.01, 100))
     cbar = plt.colorbar()
     cbar.set_label(r'$|\frac{-\overline{u^{\prime}v^{\prime}}}{\overline{\mathrm{d}U/\mathrm{d}y}}| m^2/s$', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$z/H_0$', fontsize=28)
     plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
     filename='-centerplane-computed-eddy-viscosity-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !") 

   #Surface streamlines map !
   if flagstreamsurf == 1:
     print(f"############################################################################################")
     print(f"Surface streamlines contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     Ux_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[0, Iplane]), (xinterp, yinterp), method='linear')
     Uy_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[1, Iplane]), (xinterp, yinterp), method='linear')

     ux=np.zeros((ngridy_surf,ngridx_surf))
     uy=np.zeros((ngridy_surf,ngridx_surf))
     for i in range (0,ngridy_surf):
       for j in range (0,ngridx_surf):
         c=Ux_i[i,j]
         if str(c) == '[nan]':
                ux[i,j] = 0
         else:
                ux[i,j] = c[0]
         d=Uy_i[i,j]
         if str(d) == '[nan]':
                uy[i,j] = 0
         else:
                uy[i,j] = d[0]

     #Surface streamlines contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.max(np.abs(ux))
     contour = plt.contourf(xinterp/W, yinterp/W, ux, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
     speed = np.sqrt(np.nan_to_num(ux, nan=0.0, posinf=0.0, neginf=0.0)**2 + np.nan_to_num(uy, nan=0.0, posinf=0.0, neginf=0.0)**2)
     plt.streamplot(xinterp/W, yinterp/W, ux, uy, color='black', linewidth=2*speed/speed.max())
     cbar = plt.colorbar(contour)
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagucenter == 1:
        plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if inst==0:
        cbar.set_label('$\\overline{U_x} (m/s)$', fontsize=12)
     else:
        cbar.set_label('${U_x} (m/s)$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     filename='-surfaceplane-streamlines-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Surface tke map !
   if flagtkesurf == 1 and inst != 1:
     print(f"############################################################################################")
     print(f"Surface tke contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     TKE_i = griddata((x[Iplane], y[Iplane]), TKE[Iplane], (xinterp, yinterp), method='linear')
     Ux_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[0, Iplane]), (xinterp, yinterp), method='linear')
     Uy_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[1, Iplane]), (xinterp, yinterp), method='linear')

     tke = np.zeros((ngridy_surf, ngridx_surf))
     for i in range(ngridy_surf):
       for j in range(ngridx_surf):
         c = TKE_i[i,j]
         if np.isnan(c):
             tke[i,j] = 0
         else:
             tke[i,j] = c.item()

     ux=np.zeros((ngridy_surf,ngridx_surf))
     uy=np.zeros((ngridy_surf,ngridx_surf))
     for i in range (0,ngridy_surf):
       for j in range (0,ngridx_surf):
         c=Ux_i[i,j]
         if str(c) == '[nan]':
                ux[i,j] = 0
         else:
                ux[i,j] = c[0]
         d=Uy_i[i,j]
         if str(d) == '[nan]':
                uy[i,j] = 0
         else:
                uy[i,j] = d[0]

     ux_avg=np.mean(ux) 
     uy_avg=np.mean(uy)
     
     n=25
     c=15   
     delta=5
     n = int((min(range(len(xi)), key=lambda i: abs(xi[i] - 2*W)) - c)/delta)
     jmax=np.zeros(n)
     TKEmax=np.zeros(n)
     for i in range (0,n):
        for j in range (0,len(TKE_i[:,int(i*delta+c)])):  
            if yinterp[j,int(i*delta+c)]<0:
                if TKE_i[j,int(i*delta+c)]>TKEmax[i]:
                    TKEmax[i]=TKE_i[j,int(i*delta+c)]
                    jmax[i]=j                      
     x_TKEmax=np.zeros(n+1)
     y_TKEmax=np.zeros(n+1)
     x_TKEmax[0]=0
     y_TKEmax[0]=-W
     for i in range (0,n-1):
        x_TKEmax[i+1]=xinterp[0,int(i*delta+c)]
        y_TKEmax[i+1]=yinterp[int(jmax[i]),0]
     degree=4
     func=np.polyfit(x_TKEmax,y_TKEmax,degree)
     pol = np.poly1d(func)
     yvals = pol(x_TKEmax) 
     Ub=Reb*nu/H

     #Surface tke contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(tke)/Ub**2)
     ax = plt.gca()
     ymin, ymax = ax.get_ylim()
     offset = 1.0 * (ymax - ymin)
     plt.contourf(xinterp/W, yinterp/W, tke/Ub**2, cmap=tkecolormap, levels=np.linspace(0, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        T_i = griddata((x[Iplane], y[Iplane]), T[Iplane], (xinterp, yinterp), method='linear')
        plt.contour(xinterp/W, yinterp/W, T_i/R, levels=[0.99], colors='black', linewidths=2)
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1.5)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        plt.plot(xp/W, 0, marker='*', color='black', markersize=10)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xp/W])
        ax_top.set_xticklabels(['$x_p$'],fontsize=24,color='black')
     if flagasurf == 1 and sedfoam==1:
        a_i = griddata((x[Iplane], y[Iplane]), alpha[Iplane], (xinterp, yinterp), method='linear')
        plt.contour(xinterp/W, yinterp/W, a_i, levels=[0.99 * a0], colors='black', linewidths=2)
        plt.plot(xp / W, 0, marker='*', color='black', markersize=10)
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1.5)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xp/W])
        ax_top.set_xticklabels(['$x_p$'],fontsize=24,color='black')
     ycenterline = np.argmin(np.abs(yinterp[:,0])) 
     uxcenterline = ux[ycenterline, :] / Ub
     xcenterline = xinterp[ycenterline, :] / W
     sign_change_index = np.where((uxcenterline[:-1] > 0) & (uxcenterline[1:] < 0))[0]
     if sign_change_index.size > 0:
        xcrosssign = xcenterline[sign_change_index[0] + 1]
        plt.axvline(x=xcrosssign, color='black', linestyle='-.', linewidth=1.5)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xcrosssign])
        ax_top.set_xticklabels(['$x_w$'],fontsize=24,color='black')
     cbar.set_label('$\\overline{TKE}/U_0^2$', fontsize=28)
     cbar.ax.tick_params(labelsize=24)
     quiver_spacing = 30
     q = plt.quiver(xinterp[0,::quiver_spacing]/W, yinterp[:,0][::quiver_spacing]/W, ux[::quiver_spacing,::quiver_spacing]/Ub, uy[::quiver_spacing,::quiver_spacing]/Ub, width=0.004, scale=25, headwidth=2, color='black')
     plt.tick_params(axis='x', labelsize=24) 
     plt.tick_params(axis='y', labelsize=24)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     cbar.set_ticks(np.linspace(0, +abs_max, 6))
     cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$y/W$', fontsize=28)
     stop = next((i for i, y in enumerate(yvals) if y > 0), n-1)
     stop = np.argmin(np.abs(yvals))
     #plt.scatter(x_TKEmax[1:stop:4]/W, yvals[1:stop:4]/W, color='black', marker='s', s=5)  
     #plt.scatter(x_TKEmax[1:stop:4]/W, -yvals[1:stop:4]/W, color='black', marker='s', s=5)  
     #plt.axvline(x=x_TKEmax[stop]/W, color='black', linestyle=':', linewidth=1.5)
     #plt.text(x_TKEmax[stop], ymax + offset, '$\\overline{TKE}$', ha='center', va='bottom', fontsize=5)
     filename='-surfaceplane-tke-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     print(f"############################################################################################")
     print(f"Location where distance of TKE-max from the centerline is minimum: {x_TKEmax[stop]:.5f} (m)")
     print(f"############################################################################################")

     n = 25
     c = 15
     delta = 5
     n = int((min(range(len(xi)), key=lambda i: abs(xi[i] - 2 * W)) - c) / delta)
     jfnz_down = np.zeros(n)
     jfnz_up = np.zeros(n)
     TKEfnz_down = np.zeros(n)
     TKEfnz_up = np.zeros(n)
     
     abs_max = np.nanmax(np.abs(tke))
     threshold = tkefiltercutoff * abs_max

     for i in range(n):
        for j in range(len(TKE_i[:, int(i * delta + c)])):
            if yinterp[j, int(i * delta + c)] >= 0:
                if TKE_i[j, int(i * delta + c)] >= threshold:
                    TKEfnz_down[i] = TKE_i[j, int(i * delta + c)]
                    jfnz_down[i] = j
                    break
     
     abs_max = np.nanmax(np.abs(tke))
     threshold = tkefiltercutoff * abs_max

     for i in range(n):
        for j in range(len(TKE_i[:, int(i * delta + c)]) - 1, -1, -1):
             if yinterp[j, int(i * delta + c)] >= 0:
                if TKE_i[j, int(i * delta + c)] >= threshold:
                    TKEfnz_up[i] = TKE_i[j, int(i * delta + c)]
                    jfnz_up[i] = j
                    break

     x_TKEfnz_down = np.array([xinterp[0, int(i * delta + c)] for i in range(n) if TKEfnz_down[i] > 0])
     y_TKEfnz_down = np.array([yinterp[int(jfnz_down[i]), 0] for i in range(n) if TKEfnz_down[i] > 0])
     x_TKEfnz_up = np.array([xinterp[0, int(i * delta + c)] for i in range(n) if TKEfnz_up[i] > 0])
     y_TKEfnz_up = np.array([yinterp[int(jfnz_up[i]), 0] for i in range(n) if TKEfnz_up[i] > 0])
     degree = 4
     func_down = np.polyfit(x_TKEfnz_down, y_TKEfnz_down, degree)
     pol_down = np.poly1d(func_down)
     func_up = np.polyfit(x_TKEfnz_up, y_TKEfnz_up, degree)
     pol_up = np.poly1d(func_up)
     yvals_down = pol_down(x_TKEfnz_down)
     yvals_up = pol_up(x_TKEfnz_up)
     
     #Surface filtered tke contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'], 'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(tke))
     threshold = tkefiltercutoff * abs_max
     filtke = np.where(np.abs(tke) < threshold, 0, tke)
     ax = plt.gca()
     ymin, ymax = ax.get_ylim()
     offset = 1.0 * (ymax - ymin)
     plt.contourf(xinterp/W, yinterp/W, filtke, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam == 0:
        T_i = griddata((x[Iplane], y[Iplane]), T[Iplane], (xinterp, yinterp), method='linear')
        plt.contour(xinterp/W, yinterp/W, T_i / R, levels=[0.99], colors='black', linewidths=2)
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam == 1:
        a_i = griddata((x[Iplane], y[Iplane]), alpha[Iplane], (xinterp, yinterp), method='linear')
        plt.contour(xinterp/W, yinterp/W, a_i, levels=[0.99 * a0], colors='black', linewidths=2)
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagucenter == 1:
        plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     cbar.set_label('$\\overline{TKE} ({m}^{2}/{s}^{2})$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0))
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     stopdown = min(np.argmin(np.abs(yvals_down)), np.argmax(yvals_down < 0))
     plt.plot(x_TKEfnz_down[:stopdown]/W, yvals_down[:stopdown]/W, color='black', linestyle='--')
     plt.plot(x_TKEfnz_down[:stopdown]/W, -yvals_down[:stopdown]/W, color='black', linestyle='--')
     plt.plot(x_TKEfnz_up[:stopdown]/W, yvals_up[:stopdown]/W, color='black', linestyle='--')
     plt.plot(x_TKEfnz_up[:stopdown]/W, -yvals_up[:stopdown]/W, color='black', linestyle='--')
     plt.scatter(x_TKEfnz_up[:stopdown]/W, y_TKEfnz_up[:stopdown]/W, color='grey', marker='o', s=0.5)
     plt.scatter(x_TKEfnz_down[:stopdown]/W, y_TKEfnz_down[:stopdown]/W, color='grey', marker='o', s=0.5)
     plt.scatter(x_TKEfnz_up[:stopdown]/W, -y_TKEfnz_up[:stopdown]/W, color='grey', marker='o', s=0.5)
     plt.scatter(x_TKEfnz_down[:stopdown]/W, -y_TKEfnz_down[:stopdown]/W, color='grey', marker='o', s=0.5)
     filename = '-surfaceplane-filtered-tke-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully!")

     thproxy = np.abs(y_TKEfnz_down[stopdown] - y_TKEfnz_up[stopdown])
     thproxynorm = thproxy / W

     print(f"############################################################################################")
     print(f"Shear layer thickness proxy at x=xp : th = {thproxy:.5f} (m)")
     print(f"Shear layer thickness proxy at x=xp normalized by channel width : th/W = {thproxynorm:.5f}")
     print(f"############################################################################################")

     #Surface tke streamwise profile !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(xinterp[int(ngridy_surf/2),1:], columns=['x']).to_csv(csvname, index=False)
     csvname ='-TKE-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(tke[int(ngridy_surf/2),1:], columns=['TKE']).to_csv(csvname, index=False)
     csvname ='-filTKE-x-y=0-z=0.csv'
     csvname = sim + csvname
     pd.DataFrame(filtke[int(ngridy_surf/2),1:], columns=['filTKE']).to_csv(csvname, index=False)
     plt.plot(xinterp[int(ngridy_surf/2),1:]/W, tke[int(ngridy_surf/2),1:], c='black')
     plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
     if flagucenter == 1:
        plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$\\overline{TKE} ({m}^{2}/{s}^{2})$', fontsize=12)
     stop = next((i for i, y in enumerate(yvals) if y > 0), n-1)
     stop = np.argmin(np.abs(yvals))
     plt.axvline(x=x_TKEmax[stop]/W, color='black', linestyle='--', linewidth=1)
     filename='-surfaceplane-streamwise-tke-profile.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Centerplane salinity map !
   if flagscenter == 1 and sedfoam==0:
     print(f"############################################################################################")
     print(f"Centerplane salinity contourmap !")
     print(f"############################################################################################")
     #Centerplane salinity contourmap !
     xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
     zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
     xinterp, zinterp = np.meshgrid(xi, zi)
     Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
     Ux_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[0, Iplane]), (xinterp, zinterp), method='linear')
     Uz_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[2, Iplane]), (xinterp, zinterp), method='linear')
     T_i = griddata((x[Iplane], z[Iplane]), T[Iplane], (xinterp, zinterp), method='linear')
      
     for i in range (0,len(T_i)):
        for j in range (0,len(T_i[0])):
           if zinterp[i,j]<(-H-xinterp[i,j]*tan(beta/180*pi)):
              T_i[i,j]=np.nan

     ux=np.zeros((ngridz_center,ngridx_center))
     for i in range (0,ngridz_center):
          for j in range (0,ngridx_center):
             c=Ux_i[i,j]
             if str(c)=='[nan]':
                ux[i,j]=0
             else:
                ux[i,j]=c[0]

     uz=np.zeros((ngridz_center,ngridx_center))
     for i in range (0,ngridz_center):
          for j in range (0,ngridx_center):
              d=Uz_i[i,j]
              if str(d)=='[nan]':
                  uz[i,j]=0
              else:
                  uz[i,j]=d[0]

     for i in range (0,len(Ux_i)):
          for j in range (0,len(Ux_i[0])):
              if zinterp[i,j]<(-H-xinterp[i,j]*tan(beta/180*pi)):
                 ux[i,j]=np.nan
                 uz[i,j]=np.nan
     ux_avg=np.nanmean(ux) 
     uz_avg=np.nanmean(uz)
     Ub=Reb*nu/H

     print(f"############################################################################################")
     print(f"Undercurrent point position based on 99%R criterion !")
     print(f"############################################################################################") 

     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     cf = plt.contour(xinterp, zinterp, T_i / R, levels=[0.99], colors='red', linewidths=2)
     paths = cf.get_paths()
     xvals = []
     def distance_to_line(x, y, x1, y1, x2, y2):
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        return numerator / denominator
     for path in paths:
      vertices = path.vertices 
      for vertex in vertices:
        xv, zv = vertex
        dist = distance_to_line(xv, zv, xs[0], ys[0], xs[1], ys[1])
        if dist > 1e-2: 
            xvals.append(xv)
     xuc = np.max(xvals)

     print(f"########################################################################################")
     print(f"Undercurrent point position based on 99%R criterion for time={timename} x_uc={xuc:.5f} m")
     print(f"########################################################################################")

     print(f"############################################################################################")
     print(f"Plunging unmixed river water surface area under 99%R plunge curve at the centerplane !")
     print(f"############################################################################################") 

     mask = T_i > 0.99 * R
     dx = (xinterpmax_center - xinterpmin_center) / (ngridx_center - 1)
     dz = (zinterpmax_center - zinterpmin_center) / (ngridz_center - 1)
     unmixed_river_area = np.sum(mask) * dx * dz

     print(f"###################################################################################################################")
     print(f"Total surface area of unmixed river that enters the lake: {unmixed_river_area:.3f} (m^2)")
     print(f"###################################################################################################################")

     #Centerplane salinity contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     for i in range(ngridz_center):
         for j in range(ngridx_center):
             if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                T_i[i, j] = 0
     plt.contourf(xinterp/W, zinterp/H, T_i/(R), cmap=scolormap, levels=np.linspace(0, np.nanmax(T_i/R), 100))
     cbar = plt.colorbar()
     def distance_to_line(x, y, x1, y1, x2, y2):
         numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
         denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
         return numerator / denominator
     Tfilt=T_i 
     for i in range(ngridz_center):
      for j in range(ngridx_center):
        dist = distance_to_line(xinterp[i, j], zinterp[i, j], xs[0], ys[0], xs[1], ys[1])
        if dist < 1e-2 or zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
            Tfilt[i, j] = 0 
     plt.contour(xinterp/W, zinterp/H, Tfilt / R, levels=[0.99], colors='red', linewidths=2)
     plt.contour(xinterp/W, zinterp/H, T_i/R, levels=[0.01], colors='black', linewidths=2) 
     # plt.contour(xinterp, zinterp, T_i/R, levels=[0.5], colors='red', linewidths=2)
     # plt.contour(xinterp, zinterp, T_i/R, levels=[0.25], colors='green', linewidths=2) 
     if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='red', linestyle='--', linewidth=1.5)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xp/W])
        ax_top.set_xticklabels(['$x_p$'],fontsize=24,color='black')
     if flagusurf == 1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xw = slope * (xcrosssign * W - xs[0]) + ys[0]
        plt.plot([xcrosssign, xcrosssign], [y_at_xw/H, 0], color='red', linestyle='-.', linewidth=1.5)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xcrosssign])
        ax_top.set_xticklabels(['$x_w$'],fontsize=24,color='black')
     if inst==0:
        cbar.set_label('$\\overline{R}$/$\\overline{{R}_{0}} \\%$', fontsize=28)
     else:
        cbar.set_label('${R}$/$\\overline{{R}_{0}}$ \\%', fontsize=28)
     cbar.ax.tick_params(labelsize=24)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     cbar.set_ticks(np.linspace(0, 1, 6))
     plt.tick_params(axis='x', labelsize=24) 
     plt.tick_params(axis='y', labelsize=24)
     def percentage(x, pos):
         return f'{x*100:.0f}%'  
     cbar.ax.yaxis.set_major_formatter(FuncFormatter(percentage))
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$z/H_0$', fontsize=28)
     plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
     # black_line = mlines.Line2D([], [], color='black', label='$99\\%R$', linewidth=2)
     # red_line = mlines.Line2D([], [], color='red', label='$50\\%R$', linewidth=2)
     # green_line = mlines.Line2D([], [], color='green', label='$25\\%R$', linewidth=2)
     # blue_line = mlines.Line2D([], [], color='blue', label='$10\\%R$', linewidth=2)
     # plt.legend(handles=[black_line, red_line, green_line, blue_line], loc='upper right', fontsize=6)
     filename='-centerplane-salinity-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Centerplane unmixed river water map !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     xs = np.array([0, xinterpmax_center])
     ys = np.array([-H, -H - xinterpmax_center * tan(beta / 180 * pi)])
     plt.plot(xs/W, ys/H, c='black')
     for i in range(ngridz_center):
          for j in range(ngridx_center):
              if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                  T_i[i, j] = 0 
     plt.contourf(xinterp/W, zinterp/H, np.ma.masked_where(~mask, T_i) / R, cmap=scolormap, levels=np.linspace(0, np.nanmax(T_i / R), 100))
     cbar = plt.colorbar()
     plt.contour(xinterp/W, zinterp/H, Tfilt / R, levels=[0.99], colors='red', linewidths=2)
     plt.contour(xinterp/W, zinterp/H, T_i/R, levels=[0.01], colors='black', linewidths=2)
     slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
     y_at_xuc = slope * (xuc - xs[0]) + ys[0]
     plt.plot([xuc/W, xuc/W], [y_at_xuc/H, 0], color='black', linestyle='-.', linewidth=1)
     if flagssurf == 1 and sedfoam == 0:
          slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
          y_at_xp = slope * (xp - xs[0]) + ys[0]
          plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if flagucenter == 1:
          slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
          y_at_xd = slope * (xd - xs[0]) + ys[0]
          plt.plot([xd/W, xd/W], [y_at_xd/H, 0], color='black', linestyle=':', linewidth=1)
     if inst == 0:
          cbar.set_label('$\\overline{R}$/${R}_{0}$', fontsize=28)
     else:
          cbar.set_label('${R}$/${R}_{0}$', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0))
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$z/H_0$', fontsize=28)
     plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
     filename = '-centerplane-unmixed-river-water.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Centerplane sediment concentration map !
   if flagacenter == 1 and sedfoam==1:
     print(f"############################################################################################")
     print(f"Centerplane sediment concentration contourmap !")
     print(f"############################################################################################")
     #Centerplane salinity contourmap !
     xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
     zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
     xinterp, zinterp = np.meshgrid(xi, zi)
     Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
     a_i = griddata((x[Iplane], z[Iplane]), alpha[Iplane], (xinterp, zinterp), method='linear')

     print(f"############################################################################################")
     print(f"Undercurrent point position based on 99%R criterion !")
     print(f"############################################################################################") 

     threshold = 0.99
     tolerance = 1e-4
     mask = np.abs(a_i / a0 - threshold) < tolerance
     xvals = xinterp[mask]
     if xvals.size == 0:
        xuc = 0
     else:
        xuc = np.max(xvals)

     print(f"########################################################################################")
     print(f"Undercurrent point position based on 99%R criterion for time={timename} x_uc={xuc:.5f} m")
     print(f"########################################################################################")

     #Centerplane sediment concentration field !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     for i in range(ngridz_center):
         for j in range(ngridx_center):
             if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                a_i[i, j] = 0 
     plt.contourf(xinterp/W, zinterp/H, a_i/a0, cmap=scolormap, levels=np.linspace(0, 1, 100)) 
     cbar = plt.colorbar()
     #plt.contour(xinterp/W, zinterp/H, a_i/a0, levels=[0.99], colors='black', linewidths=2) 
     #plt.contour(xinterp/W, zinterp/H, a_i, levels=[0.5*a0], colors='red', linewidths=2)
     #plt.contour(xinterp/W, zinterp/H, a_i, levels=[0.25*a0], colors='green', linewidths=2)
     #plt.contour(xinterp/W, zinterp/H, a_i, levels=[0.10*a0], colors='blue', linewidths=2)
     #plt.contour(xinterp/W, zinterp/H, a_i/a0, levels=[0.01], colors='white', linewidths=2)

     if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xsdrop - xs[0]) + ys[0]
        plt.plot([xsdrop/W, xsdrop/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=2)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xsdrop/W])
        ax_top.set_xticklabels(['$x_s$'],fontsize=24,color='black')
        #plt.suptitle(f'xp = {xp:.2f} m', fontsize=12, verticalalignment='center', horizontalalignment='center')
     if flagusurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xw = slope * (xcrosssign*W - xs[0]) + ys[0]
        plt.plot([xcrosssign, xcrosssign], [y_at_xw/H, 0], color='white', linestyle='-.', linewidth=2)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xcrosssign])
        ax_top.set_xticklabels(['$x_w$'],fontsize=24,color='black')
     if inst==0:
        cbar.set_label('$\\overline{\\alpha}/a_0$', fontsize=28)
     else: 
        cbar.set_label('${\\alpha}/a_0$', fontsize=28)
     cbar.ax.tick_params(labelsize=24)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$z/H_0$', fontsize=28)
     plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
     plt.tick_params(axis='x', labelsize=24)  
     plt.tick_params(axis='y', labelsize=24)
     # black_line = mlines.Line2D([], [], color='black', label='$99\\%\\alpha_0$', linewidth=2)
     # red_line = mlines.Line2D([], [], color='red', label='$50\\%\\alpha_0$', linewidth=2)
     # green_line = mlines.Line2D([], [], color='green', label='$25\\%\\alpha_0$', linewidth=2)
     # blue_line = mlines.Line2D([], [], color='blue', label='$10\\%\\alpha_0$', linewidth=2)
     # white_line = mlines.Line2D([], [], color='white', label='$1\\%\\alpha_0$', linewidth=2)
     # plt.legend(handles=[black_line, red_line, green_line, blue_line], loc='upper right', fontsize=6)
     filename='-centerplane-sediment-concentration-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Centerplane sediment concentration vertical profile at x=0 !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-z-x=0-y=0.csv'
     csvname = sim + csvname
     pd.DataFrame(zinterp[:-1,0]/H, columns=['z']).to_csv(csvname, index=False)
     csvname ='-alpha-z-x=0-y=0.csv'
     csvname = sim + csvname
     pd.DataFrame(a_i[:-1,0], columns=['alpha']).to_csv(csvname, index=False)
     plt.plot(a_i[:-1,0]/a0, zinterp[:-1,0]/H, c='black')
     plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     mask = (zinterp[:-1,0]/H >= (-H-0*np.tan(beta/180*np.pi))/H) & (zinterp[:-1,0]/H <= 0)
     z_profile = zinterp[:-1,0][mask]
     a_profile = a_i[:-1,0][mask]/a0
     amouth = np.trapz(a_profile, z_profile) / (z_profile[-1] - z_profile[0])
     plt.axvline(x=amouth, color='black', linestyle='--', linewidth=1)
     plt.ylim((-H-0*tan(beta/180*pi))/H,0)
     plt.ylabel('$z/H_0$', fontsize=12)
     if inst==0:
      plt.xlabel('$\\overline{a}/a_0$', fontsize=12)
     else: 
      plt.xlabel('${a}/a_0$', fontsize=12)
     filename='-centerplane-vertical-alpha-profile-at-x=0.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     print(f"##################################################################")
     print(f'Average sediment concentration at the channel outlet: {amouth:.6f}')
     print(f"##################################################################")

   #Centerplane restress map !
   if flagrecenter == 1 and inst != 1:
     print(f"############################################################################################")
     print(f"Centerplane restress contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
     zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
     xinterp, zinterp = np.meshgrid(xi, zi)
     Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
     Rxz_i = griddata((x[Iplane], z[Iplane]), Rxz[Iplane], (xinterp, zinterp), method='linear')
      
     rxz = np.zeros((ngridz_center, ngridx_center))
     for i in range(ngridz_center):
       for j in range(ngridx_center):
         c = Rxz_i[i,j]
         if np.isnan(c):
             rxz[i,j] = 0
         else:
             rxz[i,j] = c.item()

     #Centerplane restress contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     for i in range(ngridz_center):
         for j in range(ngridx_center):
             if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                rxz[i, j] = 0
     abs_max = np.nanmax(np.abs(rxz))
     plt.contourf(xinterp/W, zinterp/H, rxz, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     cbar.set_label('$\\overline{R_{xz}} ({Pa})$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$z/H_0$', fontsize=12)
     filename='-centerplane-re-stress-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Centerplane tke map !
   if flagtkecenter == 1 and inst != 1:
     print(f"############################################################################################")
     print(f"Centerplane tke contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
     zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
     xinterp, zinterp = np.meshgrid(xi, zi)
     Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
     TKE_i = griddata((x[Iplane], z[Iplane]), TKE[Iplane], (xinterp, zinterp), method='linear')
      
     tke = np.zeros((ngridz_center, ngridx_center))
     for i in range(ngridz_center):
       for j in range(ngridx_center):
         c = TKE_i[i,j]
         if np.isnan(c):
             tke[i,j] = 0
         else:
             tke[i,j] = c.item()
  
     #Centerplane tke contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     abs_max = np.nanmax(np.abs(tke))
     for i in range(ngridz_center):
         for j in range(ngridx_center):
             if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                tke[i, j] = 0 
     plt.contourf(xinterp/W, zinterp/H, tke, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if flagucenter == 1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xd = slope * (xd - xs[0]) + ys[0]
        plt.plot([xd/W, xd/W], [y_at_xd/H, 0], color='black', linestyle=':', linewidth=1)
     cbar.set_label('$\\overline{TKE} ({m}^{2}/{s}^{2})$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$z/H_0$', fontsize=12)
     filename='-centerplane-tke-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !") 

   #Centerplane vorticity map !
   if flagvortcenter == 1:
     print(f"############################################################################################")
     print(f"Centerplane vorticity contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
     zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
     xinterp, zinterp = np.meshgrid(xi, zi)
     Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
     CURLx_i = griddata((x[Iplane], z[Iplane]), np.transpose(CURL[0, Iplane]), (xinterp, zinterp), method='linear')
     CURLy_i = griddata((x[Iplane], z[Iplane]), np.transpose(CURL[1, Iplane]), (xinterp, zinterp), method='linear')
     CURLz_i = griddata((x[Iplane], z[Iplane]), np.transpose(CURL[2, Iplane]), (xinterp, zinterp), method='linear')

     curlx = np.zeros((ngridz_center, ngridx_center))
     curly = np.zeros((ngridz_center, ngridx_center))
     curlz = np.zeros((ngridz_center, ngridx_center))
     for i in range(ngridz_center):
       for j in range(ngridx_center):
         c = CURLx_i[i,j]
         d = CURLy_i[i,j]
         e = CURLz_i[i,j]
         if np.isnan(c):
             curlx[i,j] = 0
         else:
             curlx[i,j] = c.item()
         if np.isnan(d):
             curly[i,j] = 0
         else:
             curly[i,j] = d.item()
         if np.isnan(e):
             curlz[i,j] = 0
         else:
             curlz[i,j] = e.item()

     #Centerplane spanwise vorticity contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     abs_max = np.nanmax(np.abs(curly))
     for i in range(ngridz_center):
         for j in range(ngridx_center):
             if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                curly[i, j] = 0 
     plt.contourf(xinterp/W, zinterp/H, curly, cmap=vcolormap, levels=np.linspace(-abs_max/10, +abs_max/10, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if inst == 1:
        cbar.set_label('${\\omega_y} (1/{s}^{2})$', fontsize=28)
     else:
        cbar.set_label('$\\overline{{\\omega_y}} (1/{s}^{2})$', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$z/H_0$', fontsize=28)
     plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
     filename='-centerplane-spanwise-vorticity-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !") 

   #Centerplane Q-criterion map !
   if flagqcritcenter == 1:
     print(f"############################################################################################")
     print(f"Centerplane pressure contourmap !")
     print(f"############################################################################################")
     #Centerplane Q-criterion contourmap !
     xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
     zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
     xinterp, zinterp = np.meshgrid(xi, zi)
     Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
     Qcrit_i = griddata((x[Iplane], z[Iplane]), Qcrit[Iplane], (xinterp, zinterp), method='linear')

     #Centerplane Q-criterion contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     print("NaN values in p:", np.isnan(Qcrit_i).any())
     print("Inf values in p:", np.isinf(Qcrit_i).any())
     abs_max = np.nanmax(np.abs(Qcrit_i))
     for i in range(ngridz_center):
         for j in range(ngridx_center):
             if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                Qcrit_i[i, j] = 0 
     plt.contourf(xinterp/W, zinterp/H, Qcrit_i, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if inst == 1:
        cbar.set_label('$Q = \\frac{1}{2} \\left( ||\\Omega||^2 - ||S||^2 \\right)$', fontsize=28)
     else:
        cbar.set_label('$\\overline{Q} = \\frac{1}{2} \\left( ||\\overline{\\Omega}||^2 - ||\\overline{S}||^2 \\right)$', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$z/H_0$', fontsize=28)
     plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
     filename='-centerplane-Qcriterion-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")


   #Centerplane pressure map !
   if flagpcenter == 1 and inst != 0:
     print(f"############################################################################################")
     print(f"Centerplane pressure contourmap !")
     print(f"############################################################################################")
     #Centerplane pressure contourmap !
     xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
     zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
     xinterp, zinterp = np.meshgrid(xi, zi)
     Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
     p_i = griddata((x[Iplane], z[Iplane]), p[Iplane], (xinterp, zinterp), method='linear')

     #Centerplane pressure contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     print("NaN values in p:", np.isnan(p_i).any())
     print("Inf values in p:", np.isinf(p_i).any())
     abs_max = np.nanmax(np.abs(p_i))
     for i in range(ngridz_center):
         for j in range(ngridx_center):
             if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                p_i[i, j] = 0 
     plt.contourf(xinterp/W, zinterp/H, p_i, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     cbar.set_label('${p}$ (Pa)', fontsize=28)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=28)
     plt.ylabel('$z/H$', fontsize=28)
     plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
     filename='-centerplane-pressure-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !") 

   #Centerplane particle pressure map !
   if flagpacenter == 1 and inst != 0 and sedfoam==1:
     print(f"############################################################################################")
     print(f"Centerplane particle pressure contourmap !")
     print(f"############################################################################################")
     #Centerplane particle pressure contourmap !
     xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
     zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
     xinterp, zinterp = np.meshgrid(xi, zi)
     Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
     pa_i = griddata((x[Iplane], z[Iplane]), pa[Iplane], (xinterp, zinterp), method='linear')

     #Centerplane particle pressure contourmap !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'           
     xs=np.array([0, xinterpmax_center])
     ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
     plt.plot(xs/W,ys/H,c='black')
     print("NaN values in pa:", np.isnan(pa_i).any())
     print("Inf values in pa:", np.isinf(pa_i).any())
     for i in range(ngridz_center):
         for j in range(ngridx_center):
             if zinterp[i, j] < ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (xinterp[i, j] - xs[0]):
                pa_i[i, j] = 0 
     abs_max = np.nanmax(np.abs(pa_i))
     plt.contourf(xinterp/W, zinterp/H, pa_i, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1)
     cbar.set_label('${pa}$ (Pa)', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$z/H_0$', fontsize=12)
     filename='-centerplane-particle-pressure-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Centerplane shear map !
   if flagshearcenter == 1:
      print(f"############################################################################################")
      print(f"Centerplane shear stress contourmap !")
      print(f"############################################################################################")
      #Centerplane shear stress contourmap !
      xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
      zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
      xinterp, zinterp = np.meshgrid(xi, zi)
      Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
      Ux_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[0, Iplane]), (xinterp, zinterp), method='linear')
      Uz_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[2, Iplane]), (xinterp, zinterp), method='linear')

      ux=np.zeros((ngridz_center,ngridx_center))
      for i in range (0,ngridz_center):
          for j in range (0,ngridx_center):
             c=Ux_i[i,j]
             if str(c)=='[nan]':
                ux[i,j]=0
             else:
                ux[i,j]=c[0]

      uz=np.zeros((ngridz_center,ngridx_center))
      for i in range (0,ngridz_center):
          for j in range (0,ngridx_center):
              d=Uz_i[i,j]
              if str(d)=='[nan]':
                  uz[i,j]=0
              else:
                  uz[i,j]=d[0]

      for i in range (0,len(Ux_i)):
          for j in range (0,len(Ux_i[0])):
              if zinterp[i,j]<(-H-xinterp[i,j]*tan(beta/180*pi)):
                 ux[i,j]=np.nan
                 uz[i,j]=np.nan

      ux_avg=np.nanmean(ux) 
      uz_avg=np.nanmean(uz)
      Ub=Reb*nu/H

      tauxz = np.zeros_like(ux)
      for i in range(1, ngridz_center - 1):  
          for j in range(1, ngridx_center - 1):
              dux_dz = (ux[i+1, j] - ux[i-1, j]) / (zi[i+1] - zi[i-1]) 
              tauxz[i, j] = nu * rho * dux_dz
      tauxz[np.isnan(ux)] = np.nan
      
      ut = (ux**2 + uz**2)**0.5
      epsilon = 1e-12 
      gamma = np.arctan(uz / (ux + epsilon)) 
      alphagon = beta/180*pi - gamma 
      us = np.cos(alphagon)*ut
      un = np.sin(alphagon)*ut
      us[ux < 0] = -np.abs(us[ux < 0])

      us[np.isnan(us) | np.isinf(us)] = 0
      un[np.isnan(un) | np.isinf(un)] = 0

      tausn = np.zeros_like(us)
      dus_dx = np.zeros_like(us)
      dus_dz = np.zeros_like(us)

      for i in range(1, ngridz_center - 1):
        for j in range(1, ngridx_center - 1):
            dus_dx[i, j] = (us[i, j+1] - us[i, j-1]) / (xi[j+1] - xi[j-1])
            dus_dz[i, j] = (us[i+1, j] - us[i-1, j]) / (zi[i+1] - zi[i-1])

      cosbeta = np.cos(beta / 180 * np.pi)
      sinbeta = np.sin(beta / 180 * np.pi)
      dus_dn = dus_dx * sinbeta + dus_dz * cosbeta 
      tausn = nu * rho * dus_dn
      tausn[np.isnan(us)] = np.nan

      tausnbed = np.zeros(ngridx_center)
      s = -H - xinterp[0, :] * np.tan(beta / 180 * np.pi) + 0.5 * H
      for j in range(ngridx_center):
        closest_index = np.argmin(np.abs(zinterp[:, j] - s[j]))
        tausnbed[j] = tausn[closest_index, j]

      print(f"############################################################################################")
      print(f"Undercurrent point position based on bed shear stress criterion !")
      print(f"############################################################################################") 
      
      xvaluctau = xinterp[0,:]
      tausnbed = tausnbed[:]
      tausnbed = tausnbed[::-1]
      xvaluctau = xvaluctau[::-1]
      min_index = np.argmin(tausnbed)
      xucmin = xvaluctau[min_index]
      tausn_min = tausnbed[min_index]
      crossings_before_min = np.where(np.diff(np.sign(tausnbed[min_index:])))[0]
      if crossings_before_min.size > 0:
        xuctau = xvaluctau[min_index + crossings_before_min[0]]
      else:
        xuctau = None
      tausnbed = tausnbed[::-1]
      xvaluctau = xvaluctau[::-1]

      print(f"######################################################################################################################################################################################################")
      print(f"Undercurrent point position based on bed shear stress at n=+H/2 over bed level criterion for time={timename} x_uc_tau={xuctau:.5f} m")
      print(f"Undercurrent point position based on bed shear stress at n=+H/2 over bed  level criterion for time={timename} normalized by the channel width x_uc_tau/W={xuctau/W:.5f} m")
      print(f"Undercurrent minimum negative bed shear stress position based on bed shear stress at n=+H/2 over bed level criterion for time={timename} x_uc_tau_min={xucmin:.5f} m")
      print(f"Undercurrent minimum negative bed shear stress position based on bed shear stress at n=+H/2 over bed level criterion for time={timename} normalized by the channel width x_uc_tau_min/W={xucmin/W:.5f}")
      print(f"######################################################################################################################################################################################################")
      
      #Shear stress tauxz component centerplane contourmap !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'          
      xs=np.array([0, xinterpmax_center])
      ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
      plt.plot(xs/W,ys/H,c='black') 
      abs_max = np.nanmax(np.abs(tauxz/(rho*Ub**2)))
      contour = plt.contourf(xinterp/W, zinterp/H, tauxz/(rho*Ub**2), cmap=vcolormap, levels=np.linspace(-0.25*abs_max, 0.25*abs_max, 100))
      if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1.5)
      if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1.5) 
      if flagucenter == 1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xd = slope * (xd - xs[0]) + ys[0]
        plt.plot([xd/W, xd/W], [y_at_xd/H, 0], color='black', linestyle='-.', linewidth=1.5)
      if flagscenter == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xuc = slope * (xuc - xs[0]) + ys[0]
        plt.plot([xuc/W, xuc/W], [y_at_xuc/H, 0], color='black', linestyle=':', linewidth=1.5)
      cbar = plt.colorbar(contour)
      if inst==0:
         cbar.set_label('$\\overline{\\tau_{xz}}/\\rho U_0^2$', fontsize=28)
      else:
         cbar.set_label('${\\tau_{xz}} /\\rho U_0^2$', fontsize=28)
      cbar.ax.tick_params(labelsize=24)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      cbar.set_ticks(np.linspace(-0.25*abs_max, +0.25*abs_max, 6))
      plt.tick_params(axis='x', labelsize=24) 
      plt.tick_params(axis='y', labelsize=24)
      plt.xlabel('$x/W$', fontsize=28)
      plt.ylabel('$z/H_0$', fontsize=28)
      plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
      filename='-centerplane-shear-tauxz-stress-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Shear stress tausn component centerplane contourmap !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'          
      xs=np.array([0, xinterpmax_center])
      ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
      yshigher=np.array([-H+0.5*H,-H-xinterpmax_center*tan(beta/180*pi)+0.5*H])
      plt.plot(xs/W,ys/H,c='black') 
      plt.plot(xs/W,yshigher/H,c='black',linestyle='--')
      abs_max = np.nanmax(np.abs(tausn/(rho*Ub**2)))
      contour = plt.contourf(xinterp/W, zinterp/H, tausn /(rho*Ub**2), cmap=vcolormap, levels=np.linspace(-0.35*abs_max, +0.35*abs_max, 100))
      if flagssurf == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1.5)
      if flagasurf == 1 and sedfoam==1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xp = slope * (xp - xs[0]) + ys[0]
        plt.plot([xp/W, xp/W], [y_at_xp/H, 0], color='black', linestyle='--', linewidth=1.5)  
      if flagucenter == 1:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xd = slope * (xd - xs[0]) + ys[0]
        plt.plot([xd/W, xd/W], [y_at_xd/H, 0], color='black', linestyle='-.', linewidth=1.5)
      if flagscenter == 1 and sedfoam==0:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y_at_xuc = slope * (xuc - xs[0]) + ys[0]
        plt.plot([xuc/W, xuc/W], [y_at_xuc/H, 0], color='black', linestyle=':', linewidth=1.5)
      cbar = plt.colorbar(contour)
      if inst==0:
         cbar.set_label('$\\overline{\\tau_{sn}}/\\rho U_0^2$', fontsize=28)
      else:
         cbar.set_label('${\\tau_{sn}} /\\rho U_0^2$', fontsize=28)
      cbar.ax.tick_params(labelsize=16) 
      cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      cbar.set_ticks(np.linspace(-0.35*abs_max, +0.35*abs_max, 6))
      plt.tick_params(axis='x', labelsize=24) 
      plt.tick_params(axis='y', labelsize=24)
      plt.xlabel('$x/W$', fontsize=28)
      plt.ylabel('$z/H_0$', fontsize=28)
      plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
      filename='-centerplane-shear-tausn-stress-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Shear stress tausn component bed profile !
      fig = plt.figure(figsize=(6, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      csvname ='-x-bed-shear-stress.csv'
      csvname = sim + csvname
      pd.DataFrame(xinterp[0,:], columns=['x']).to_csv(csvname, index=False)
      csvname ='-tausn-bed-shear-stress.csv'
      csvname = sim + csvname
      pd.DataFrame(tausnbed, columns=['tausn']).to_csv(csvname, index=False)
      plt.plot(xinterp[0,:]/W, tausnbed, color='black', label=r'$\tau_{sn}(x)$')
      plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
      plt.axvline(x=xuctau/W, color='black', linestyle='--', linewidth=1)
      plt.xlabel('$x/W$', fontsize=12)
      if inst==0:
          plt.ylabel('$\\overline{\\tau_{sn}} (Pa)$', fontsize=12)
      else:
          plt.ylabel('${\\tau_{sn}} (Pa)$', fontsize=12)
      plt.legend(loc='upper right')
      filename='-bed-shear-tausn-stress-profile.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

   #Ep coefficient !
   if flagep == 1:
      print(f"############################################################################################")
      print(f"Ep(x) - entrainement coefficient curve !")
      print(f"############################################################################################")
      yi = np.linspace(yinterpmin_channel, yinterpmax_channel, ngridy_channel)
      zi = np.linspace(zinterpmin_channel, zinterpmax_channel, ngridz_channel)
      yinterp, zinterp = np.meshgrid(yi, zi)
      Iplane = np.where(np.logical_and(x >= Xplane_channel - dx_channel, x <= Xplane_channel + dx_channel))
      original_shape = U.shape
      new_size = max(Iplane[0]) + 1  
      if original_shape[1] <= new_size:
         U_expanded = np.zeros((original_shape[0], new_size))
         U_expanded[:, :original_shape[1]] = U
      else:
         U_expanded = U
      Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
      ux_channel = np.zeros((ngridz_channel, ngridy_channel))
      for i in range(ngridz_channel):
         for j in range(ngridy_channel):
             c = Ux_i[i,j]
             if str(c) == '[nan]':
                ux_channel[i,j] = 0
             else:
                ux_channel[i,j] = c[0]  
      Ub_channel = np.mean(ux_channel)
      Q0_channel=Ub_channel*H*W
      Q0_nominal=Reb*nu*W
    
      yi = np.linspace(yinterpmin_ep, yinterpmax_ep, ngridy_ep)
      zi = np.linspace(zinterpmin_ep, zinterpmax_ep, ngridz_ep)
      yinterp, zinterp = np.meshgrid(yi, zi)
      Ep=np.zeros(int(Nxc*Lxb))
      xc=np.zeros(int(Nxc*Lxb))
      Umplume=np.zeros(int(Nxc*Lxb))
      Acsave=np.zeros(int(Nxc*Lxb))
      m=0
      for n in range (0,int(Nxc*Lxb)+1,step):
          xc[m]=(1/Nxc)*n
          Xplane_ep=(1/Nxc)*n
          Iplane=np.where(np.logical_and(x>=Xplane_ep-dx_ep,x<=Xplane_ep+dx_ep))
          original_shape = U.shape
          new_size = max(Iplane[0]) + 1  
          if original_shape[1] <= new_size:
             U_expanded = np.zeros((original_shape[0], new_size))
             U_expanded[:, :original_shape[1]] = U
          else:
             U_expanded = U
          Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')

          if sedfoam==0:
              original_shape = T.shape
              new_size = max(Iplane[0]) + 1  
              if original_shape[0] <= new_size:
                 T_expanded = np.zeros(new_size)
                 T_expanded[:original_shape[0]] = T
              else:
                 T_expanded = T
              T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')

              Ac=0
              thres=0.005*R
              u=np.zeros((ngridz_ep,ngridy_ep))
              for i in range (0,ngridz_ep):
                for j in range (0,ngridy_ep):
                    value = Ux_i[i, j].item()  
                    if T_i[i,j].item() > thres:
                       u[i,j] = value
                       Ac+=Lyb/ngridy_ep*Lzb/ngridz_ep
                    else:
                       u[i,j] = 0
          else:
              original_shape = alpha.shape
              new_size = max(Iplane[0]) + 1  
              if original_shape[0] <= new_size:
                 alpha_expanded = np.zeros(new_size)
                 alpha_expanded[:original_shape[0]] = alpha
              else:
                 alpha_expanded = alpha
              alpha_i = griddata((y[Iplane], z[Iplane]), alpha_expanded[Iplane], (yinterp, zinterp), method='linear')

              Ac=0
              thres=0.005*a0
              u=np.zeros((ngridz_ep,ngridy_ep))
              for i in range (0,ngridz_ep):
                for j in range (0,ngridy_ep):
                    value = Ux_i[i, j].item()  
                    if alpha_i[i,j].item() > thres:
                       u[i,j] = value
                       Ac+=Lyb/ngridy_ep*Lzb/ngridz_ep
                    else:
                       u[i,j] = 0

          Q=0
          for i in range (0,ngridz_ep):
            for j in range (0,ngridy_ep):
                Q+=u[i,j]*Lyb/ngridy_ep*Lzb/ngridz_ep
          if n==0:
             Q0=Q  
             Ac0=Ac    
          Ep[m]=(Q-Q0)/(Q0)
          Umplume[m]=Q/Ac
          Acsave[m]=Ac
          m=m+1
      
      max_Ep = np.max(Ep)
      max_Ep_index = np.argmax(Ep)
      print(f"####################################################################")
      print(f"Channel nominal flowrate: Q0={Q0_nominal:.5f} m^3/s")
      print(f"Channel inlet flowrate: Q0_channel={Q0_channel:.5f} m^3/s")
      print(f"Channel outlet flowrate: Q0={Q0:.5f} m^3/s")
      print(f"Channel outlet crossection: Ac0={Ac0:.5f} m^2")
      print(f"Channel outlet mean plume velocity: Umplume0={Umplume[0]:.5f} m/s")
      print(f"Entrainement coefficient computation total number of iterations: {n}")
      print(f"Entrainement coefficient maximum: {max_Ep:.5f}")
      print(f"Entrainement coefficient maximum location: {xc[max_Ep_index]:.5f}")
      print(f"####################################################################")

      #Ep(x) curve !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'           
      csvname ='-x-coordinate-Ep.csv'
      csvname = sim + csvname
      pd.DataFrame(xc[:m-1], columns=['x']).to_csv(csvname, index=False)
      csvname ='-Ep-coordinate-Ep.csv'
      csvname = sim + csvname
      pd.DataFrame(Ep[:m-1], columns=['Ep']).to_csv(csvname, index=False)
      plt.plot(xc[:m-1]/W, Ep[:m-1], '-', linewidth=2, color='black', label=f'{sim}')
      if flagucenter == 1:
         plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
         plt.axvline(x=xumin/W, color='black', linestyle='--', linewidth=1)
      if flagssurf == 1 and sedfoam==0:
         plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
      if flagasurf == 1 and sedfoam==1:
         plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
      plt.xlabel('$x/W$', fontsize=12)
      plt.ylabel('$E_p$', fontsize=12)
      filename='-curve-entrainement-coefficient.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !") 

      #Mean velocity of the plume through the cross-section Umplume(x) !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      csvname = '-x-mean-plume-velocity-through-crossection.csv'
      csvname = sim + csvname
      pd.DataFrame(xc[:m-1], columns=['x']).to_csv(csvname, index=False)
      csvname = '-Umplume-mean-plume-velocity-through-crossection.csv'
      csvname = sim + csvname
      pd.DataFrame(Umplume[:m-1], columns=['Umplume']).to_csv(csvname, index=False)
      plt.plot(xc[:m-1]/W, Umplume[:m-1], '-', linewidth=2, color='black', label=f'{sim}')
      if flagucenter == 1:
        plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
        plt.axvline(x=xumin/W, color='black', linestyle='--', linewidth=1)
      if flagssurf == 1 and sedfoam == 0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
      if flagasurf == 1 and sedfoam == 1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
      plt.xlabel('$x/W$', fontsize=12)
      plt.ylabel(r'$U_{mplume}(x)$ (m/s)', fontsize=12)
      filename = '-Umplume-mean-plume-velocity-through-crossection.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f'Plotted "{savename}" successfully!')

      #Cross-sectional plume area Ac(x) !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      csvname = '-x-plume-cross-sectional-area.csv'
      csvname = sim + csvname
      pd.DataFrame(xc[:m-1], columns=['x']).to_csv(csvname, index=False)
      csvname = '-Ac-plume-cross-sectional-area.csv'
      csvname = sim + csvname
      pd.DataFrame(Acsave[:m-1], columns=['Ac']).to_csv(csvname, index=False)
      plt.plot(xc[:m-1]/W, Acsave[:m-1], '-', linewidth=2, color='black', label=f'{sim}')
      if flagucenter == 1:
        plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
        plt.axvline(x=xumin/W, color='black', linestyle='--', linewidth=1)
      if flagssurf == 1 and sedfoam == 0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
      if flagasurf == 1 and sedfoam == 1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
      plt.xlabel('$x/W$', fontsize=12)
      plt.ylabel(r'$A_c(x)$ (m$^2$)', fontsize=12)
      filename = '-Ac-plume-cross-sectional-area.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f'Plotted "{savename}" successfully!')

   #Mass conservation -> (Qout(x)-Q0)/Q0 check !
   if flagqout == 1:
      print(f"############################################################################################")
      print(f"(Qout(x)-Q0)/Q0 - curve !")
      print(f"############################################################################################")
      yi = np.linspace(yinterpmin_channel, yinterpmax_channel, ngridy_channel)
      zi = np.linspace(zinterpmin_channel, zinterpmax_channel, ngridz_channel)
      yinterp, zinterp = np.meshgrid(yi, zi)
      Iplane = np.where(np.logical_and(x >= Xplane_channel - dx_channel, x <= Xplane_channel + dx_channel))
      original_shape = U.shape
      new_size = max(Iplane[0]) + 1  
      if original_shape[1] <= new_size:
         U_expanded = np.zeros((original_shape[0], new_size))
         U_expanded[:, :original_shape[1]] = U
      else:
         U_expanded = U
      Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
      ux_channel = np.zeros((ngridz_channel, ngridy_channel))
      for i in range(ngridz_channel):
         for j in range(ngridy_channel):
             c = Ux_i[i,j]
             if str(c) == '[nan]':
                ux_channel[i,j] = 0
             else:
                ux_channel[i,j] = c[0]  
      Ub_channel = np.mean(ux_channel)
      Q0_channel=Ub_channel*H*W
      Q0_nominal=Reb*nu*W
    
      yi = np.linspace(yinterpmin_ep, yinterpmax_ep, ngridy_ep)
      zi = np.linspace(zinterpmin_ep, zinterpmax_ep, ngridz_ep)
      yinterp, zinterp = np.meshgrid(yi, zi)
      Qout=np.zeros(int(10*Lxb))
      xc=np.zeros(int(10*Lxb))
      m=0
      for n in range (0,int(10*Lxb)+1,step):
          xc[m]=0.1*n
          Xplane_ep=0.1*n
          Iplane=np.where(np.logical_and(x>=Xplane_ep-dx_ep,x<=Xplane_ep+dx_ep))
          original_shape = U.shape
          new_size = max(Iplane[0]) + 1  
          if original_shape[1] <= new_size:
             U_expanded = np.zeros((original_shape[0], new_size))
             U_expanded[:, :original_shape[1]] = U
          else:
             U_expanded = U
          Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
          
          u=np.zeros((ngridz_ep,ngridy_ep))
          for i in range (0,ngridz_ep):
            for j in range (0,ngridy_ep):
              c=Ux_i[i,j]
              if str(c)=='[nan]':
                u[i,j]=0
              else:
                u[i,j]=c[0]

          Q=0
          for i in range (0,ngridz_ep):
            for j in range (0,ngridy_ep):
                Q+=u[i,j]*Lyb/ngridy_ep*Lzb/ngridz_ep
          if n==0:
                Q0=Q       
          Qout[m]=(Q-Q0)/(Q0)
          m=m+1
      
      max_Qout = np.max(Qout)
      max_Qout_index = np.argmax(Qout)
      print(f"###########################################################################################")
      print(f"Channel nominal flowrate: Q0={Q0_nominal:.5f} m^3/s")
      print(f"Channel inlet flowrate: Q0_channel={Q0_channel:.5f} m^3/s")
      print(f"Channel outlet flowrate: Q0={Q0:.5f} m^3/s")
      print(f"Outflow from crossection at x -> (Qout(x)-Q0)/(Q0) computation total number of iterations: {n}")
      print(f"Outflow from crossection at x -> (Qout(x)-Q0)/(Q0) maximum: {max_Qout:.5f}")
      print(f"Outflow from crossection at x -> (Qout(x)-Q0)/(Q0) maximum location: {xc[max_Qout_index]:.5f}")
      print(f"###########################################################################################")

      #Mass conservation -> (Qout(x)-Q0)/Q0 curve !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'           
      csvname ='-x-coordinate-Qout.csv'
      csvname = sim + csvname
      pd.DataFrame(xc[:m-1], columns=['x']).to_csv(csvname, index=False)
      csvname ='-Qout-coordinate-Ep.csv'
      csvname = sim + csvname
      pd.DataFrame(Qout[:m-1], columns=['Qout']).to_csv(csvname, index=False)
      plt.plot(xc[:m-1]/W, Qout[:m-1]*100, '-', linewidth=2, color='black', label=f'{sim}')
      if flagucenter == 1:
         plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
         plt.axvline(x=xumin/W, color='black', linestyle='--', linewidth=1)
      if flagssurf == 1 and sedfoam==0:
         plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
      if flagasurf == 1 and sedfoam==1:
         plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
      plt.xlabel('$x/W$', fontsize=12)
      plt.ylabel('$(Qout(x)-Q0)/Q0 \\%$', fontsize=12)
      plt.ylim(-50,50)
      filename='-curve-Qout-mass-conservation.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

   #Transversal plane salinity map !
   if flagstrans == 1 and sedfoam==0:
      print(f"############################################################################################")
      print(f"Transversalplane salinity contourmap !")
      print(f"############################################################################################")
      #Transversalplane salinity contourmap !
      yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
      zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
      yinterp, zinterp = np.meshgrid(yi, zi)
      Iplane=np.where(np.logical_and(x>=Xplane_trans-dx_trans,x<=Xplane_trans+dx_trans))
      original_shape = T.shape
      new_size = max(Iplane[0]) + 1  
      if original_shape[0] <= new_size:
         T_expanded = np.zeros(new_size)
         T_expanded[:original_shape[0]] = T
      else:
         T_expanded = T
      T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')

      #Transversalplane salinity contourmap at x=0 !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'           
      plt.contourf(yinterp/W, zinterp/H, T_i/(R), cmap=scolormap, levels=np.linspace(0, np.nanmax(T_i/R), 100))
      plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
      plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
      cbar = plt.colorbar()
      if inst==0:
         cbar.set_label('$\\overline{R}$/${R}_{0}$', fontsize=12)
      else:
         cbar.set_label('${R}$/${R}_{0}$', fontsize=12)
      cbar.ax.tick_params(labelsize=10)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.xlabel('$y/W$', fontsize=12)
      plt.ylabel('$z/H_0$', fontsize=12)
      plt.ylim(Dtr/H,0)
      filename='-transversalplane-x=0-salinity-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Transversal salinity contourmap at x=xp/2 !
      if flagssurf == 1 and sedfoam==0:
         yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=0.5*xp-dx_trans,x<=0.5*xp+dx_trans))
         original_shape = T.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[0] <= new_size:
            T_expanded = np.zeros(new_size)
            T_expanded[:original_shape[0]] = T
         else:
            T_expanded = T
         T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')
         
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'           
         plt.contourf(yinterp/W, zinterp/H, T_i/(R), cmap=scolormap, levels=np.linspace(0, np.nanmax(T_i/R), 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         if inst==0:
            cbar.set_label('$\\overline{R}$/${R}_{0}$', fontsize=12)
         else:
            cbar.set_label('${R}$/${R}_{0}$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxp=-0.5*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=0.5xp-salinity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal salinity contourmap at x=xp !
      if flagssurf == 1 and sedfoam==0:
         yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=xp-dx_trans,x<=xp+dx_trans))
         original_shape = T.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[0] <= new_size:
            T_expanded = np.zeros(new_size)
            T_expanded[:original_shape[0]] = T
         else:
            T_expanded = T
         T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')
         
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'           
         plt.contourf(yinterp/W, zinterp/H, T_i/(R), cmap=scolormap, levels=np.linspace(0, np.nanmax(T_i/R), 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         if inst==0:
            cbar.set_label('$\\overline{R}$/${R}_{0}$', fontsize=12)
         else:
            cbar.set_label('${R}$/${R}_{0}$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxp=-xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=xp-salinity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal salinity contourmap at x=2xp !
      if flagssurf == 1 and sedfoam==0:
         yi = np.linspace(2*yinterpmin_trans, 2*yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=2*xp-dx_trans,x<=2*xp+dx_trans))
         original_shape = T.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[0] <= new_size:
            T_expanded = np.zeros(new_size)
            T_expanded[:original_shape[0]] = T
         else:
            T_expanded = T
         T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')
         
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'           
         plt.contourf(yinterp/W, zinterp/H, T_i/(R), cmap=scolormap, levels=np.linspace(0, np.nanmax(T_i/R), 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         if inst==0:
            cbar.set_label('$\\overline{R}$/${R}_{0}$', fontsize=12)
         else:
            cbar.set_label('${R}$/${R}_{0}$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxp=-2*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=2xp-salinity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal salinity contourmap at x=3xp !
      if flagssurf == 1 and sedfoam==0:
         yi = np.linspace(3*yinterpmin_trans, 3*yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=3*xp-dx_trans,x<=3*xp+dx_trans))
         original_shape = T.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[0] <= new_size:
            T_expanded = np.zeros(new_size)
            T_expanded[:original_shape[0]] = T
         else:
            T_expanded = T
         T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')
         
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'           
         plt.contourf(yinterp/W, zinterp/H, T_i/(R), cmap=scolormap, levels=np.linspace(0, np.nanmax(T_i/R), 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         if inst==0:
            cbar.set_label('$\\overline{R}$/${R}_{0}$', fontsize=12)
         else:
            cbar.set_label('${R}$/${R}_{0}$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxp=-3*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=3xp-salinity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

   #Transversal plane sediment concentration map !
   if flagatrans == 1 and sedfoam==1:
      print(f"############################################################################################")
      print(f"Transversalplane sediment concentration contourmap !")
      print(f"############################################################################################")
      #Transversalplane sediment concentration contourmap !
      yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
      zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
      yinterp, zinterp = np.meshgrid(yi, zi)
      Iplane=np.where(np.logical_and(x>=Xplane_trans-dx_trans,x<=Xplane_trans+dx_trans))
      original_shape = alpha.shape
      new_size = max(Iplane[0]) + 1  
      if original_shape[0] <= new_size:
         alpha_expanded = np.zeros(new_size)
         alpha_expanded[:original_shape[0]] = alpha
      else:
         alpha_expanded=alpha
      a_i = griddata((y[Iplane], z[Iplane]), alpha_expanded[Iplane], (yinterp, zinterp), method='linear')

      #Transversalplane sediment concentration contourmap at x=0 !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in' 
      aclip = np.clip(a_i/a0, 1e-10, 0.65/a0)          
      plt.contourf(yinterp/W, zinterp/H, aclip, levels=np.logspace(-10, np.log10(0.65/a0), 100), cmap=scolormap, norm=clr.LogNorm(vmin=1e-10, vmax=0.65/a0))      
      plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
      plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
      cbar = plt.colorbar()
      if inst==0:
         cbar.set_label('$\\overline{\\alpha}/a_0$', fontsize=12)
      else:
         cbar.set_label('${\\alpha}/a_0$', fontsize=12)
      cbar.ax.tick_params(labelsize=10)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.xlabel('$y/W$', fontsize=12)
      plt.ylabel('$z/H_0$', fontsize=12)
      plt.ylim(Dtr/H,0)
      filename='-transversalplane-x=0-sediment-concentration-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Transversal sediment concentration contourmap at x=xsdrop/2 !
      if flagasurf == 1 and sedfoam == 1:
         yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=0.5*xsdrop-dx_trans,x<=0.5*xsdrop+dx_trans))
         original_shape = alpha.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[0] <= new_size:
           alpha_expanded = np.zeros(new_size)
           alpha_expanded[:original_shape[0]] = alpha
         else:
           alpha_expanded=alpha
         a_i = griddata((y[Iplane], z[Iplane]), alpha_expanded[Iplane], (yinterp, zinterp), method='linear')

         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'  
         aclip = np.clip(a_i/a0, 1e-10, 0.65/a0)          
         plt.contourf(yinterp/W, zinterp/H, aclip, levels=np.logspace(-10, np.log10(0.65/a0), 100), cmap=scolormap, norm=clr.LogNorm(vmin=1e-10, vmax=0.65/a0)) 
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         if inst==0:
            cbar.set_label('$\\overline{\\alpha}/a_0$', fontsize=12)
         else:
            cbar.set_label('$\\alpha/a_0$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxs=-0.5*xsdrop*tan(beta/180*pi)-H;
         plt.ylim(Dxs/H,0)
         filename='-transversalplane-x=0.5xp-sediment-concentration-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal sediment concentration contourmap at x=xsdrop !
      if flagasurf == 1 and sedfoam == 1:
         yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=xsdrop-dx_trans,x<=xsdrop+dx_trans))
         original_shape = alpha.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[0] <= new_size:
           alpha_expanded = np.zeros(new_size)
           alpha_expanded[:original_shape[0]] = alpha
         else:
           alpha_expanded=alpha
         a_i = griddata((y[Iplane], z[Iplane]), alpha_expanded[Iplane], (yinterp, zinterp), method='linear')

         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'  
         aclip = np.clip(a_i/a0, 1e-10, 0.65/a0)           
         plt.contourf(yinterp/W, zinterp/H, aclip, levels=np.logspace(-10, np.log10(0.65/a0), 100), cmap=scolormap, norm=clr.LogNorm(vmin=1e-10, vmax=0.65/a0)) 
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         if inst==0:
            cbar.set_label('$\\overline{\\alpha}/a_0$', fontsize=12)
         else:
            cbar.set_label('$\\alpha/a_0$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxs=-xsdrop*tan(beta/180*pi)-H;
         plt.ylim(Dxs/H,0)
         filename='-transversalplane-x=xp-sediment-concentration-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal sediment concentration contourmap at x=2xsdrop !
      if flagasurf == 1 and sedfoam == 1:
         yi = np.linspace(2*yinterpmin_trans, 2*yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=2*xsdrop-dx_trans,x<=2*xsdrop+dx_trans))
         original_shape = alpha.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[0] <= new_size:
           alpha_expanded = np.zeros(new_size)
           alpha_expanded[:original_shape[0]] = alpha
         else:
           alpha_expanded=alpha
         a_i = griddata((y[Iplane], z[Iplane]), alpha_expanded[Iplane], (yinterp, zinterp), method='linear')

         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'   
         aclip = np.clip(a_i/a0, 1e-10, 0.65/a0)          
         plt.contourf(yinterp/W, zinterp/H, aclip, levels=np.logspace(-10, np.log10(0.65/a0), 100), cmap=scolormap, norm=clr.LogNorm(vmin=1e-10, vmax=0.65/a0)) 
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         if inst==0:
            cbar.set_label('$\\overline{\\alpha}/a_0$', fontsize=12)
         else:
            cbar.set_label('$\\alpha/a_0$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxs=-2*xsdrop*tan(beta/180*pi)-H;
         plt.ylim(Dxs/H,0)
         filename='-transversalplane-x=2xp-sediment-concentration-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal sediment concentration contourmap at x=3xsdrop !
      if flagasurf == 1 and sedfoam == 1:
         yi = np.linspace(3*yinterpmin_trans, 3*yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=3*xsdrop-dx_trans,x<=3*xsdrop+dx_trans))
         original_shape = alpha.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[0] <= new_size:
           alpha_expanded = np.zeros(new_size)
           alpha_expanded[:original_shape[0]] = alpha
         else:
           alpha_expanded=alpha
         a_i = griddata((y[Iplane], z[Iplane]), alpha_expanded[Iplane], (yinterp, zinterp), method='linear')

         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         aclip = np.clip(a_i/a0, 1e-10, 0.65/a0)             
         plt.contourf(yinterp/W, zinterp/H, aclip, levels=np.logspace(-10, np.log10(0.65/a0), 100), cmap=scolormap, norm=clr.LogNorm(vmin=1e-10, vmax=0.65/a0)) 
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         if inst==0:
            cbar.set_label('$\\overline{\\alpha}/a_0$', fontsize=12)
         else:
            cbar.set_label('$\\alpha/a_0$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxs=-3*xsdrop*tan(beta/180*pi)-H;
         plt.ylim(Dxs/H,0)
         filename='-transversalplane-x=3xp-sediment-concentration-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

   #Transversalplane velocity map !
   if flagutrans == 1:
      print(f"############################################################################################")
      print(f"Transversalplane velocity  contourmap !")
      print(f"############################################################################################")
      #Transversalplane velocity contourmap !
      yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
      zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
      yinterp, zinterp = np.meshgrid(yi, zi)
      Iplane=np.where(np.logical_and(x>=Xplane_trans-dx_trans,x<=Xplane_trans+dx_trans))
      original_shape = U.shape
      new_size = max(Iplane[0]) + 1  
      if original_shape[1] <= new_size:
         U_expanded = np.zeros((original_shape[0], new_size))
         U_expanded[:, :original_shape[1]] = U
      else:
         U_expanded = U
      Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
      Uy_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[1, Iplane]), (yinterp, zinterp), method='linear')
      Uz_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[2, Iplane]), (yinterp, zinterp), method='linear')
      ux=np.zeros((ngridz_trans,ngridy_trans))
      for i in range (0,ngridz_trans):
        for j in range (0,ngridy_trans):
            c=Ux_i[i,j]
            if str(c)=='[nan]':
               ux[i,j]=0
            else:
               ux[i,j]=c[0]
      uy=np.zeros((ngridz_trans,ngridy_trans))
      for i in range (0,ngridz_trans):
        for j in range (0,ngridy_trans):
            c=Uy_i[i,j]
            if str(c)=='[nan]':
               uy[i,j]=0
            else:
               uy[i,j]=c[0]
      uz=np.zeros((ngridz_trans,ngridy_trans))
      for i in range (0,ngridz_trans):
        for j in range (0,ngridy_trans):
            c=Uz_i[i,j]
            if str(c)=='[nan]':
               uz[i,j]=0
            else:
               uz[i,j]=c[0]
      
      Ub=Reb*nu/H

      #Transversalplane streamwise velocity contourmap at x=0 !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'           
      abs_max = np.nanmax(np.abs(ux/Ub))
      contour = plt.contourf(yinterp/W, zinterp/H, ux/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
      plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
      plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
      cbar = plt.colorbar(contour)
      quiver_spacing = 8
      q = plt.quiver(yinterp[0,::quiver_spacing*2]/W, zinterp[:,0][::quiver_spacing//4]/H, uy[::quiver_spacing//4,::quiver_spacing*2]/Ub, uz[::quiver_spacing//4,::quiver_spacing*2]/Ub, width=0.002, scale=10, headwidth=2, color='black')
      if inst==0:
         cbar.set_label('$\\overline{u}/U_0$', fontsize=28)
      else:
         cbar.set_label('${u}/U_0$', fontsize=28)
      cbar.ax.tick_params(labelsize=24)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      plt.tick_params(axis='x', labelsize=24) 
      plt.tick_params(axis='y', labelsize=24)
      cbar.ax.yaxis.set_major_formatter(formatter)
      cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
      cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      plt.xlabel('$y/W$', fontsize=28)
      plt.ylabel('$z/H_0$', fontsize=28)
      plt.ylim(Dtr/H,0)
      filename='-transversalplane-x=0-velocity-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")
      
      #Transversal velocity contourmap at x=xp/2 !
      if flagssurf == 1:
         yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=0.5*xp-dx_trans,x<=0.5*xp+dx_trans))
         original_shape = U.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[1] <= new_size:
              U_expanded = np.zeros((original_shape[0], new_size))
              U_expanded[:, :original_shape[1]] = U
         else:
              U_expanded = U
         Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
         Uy_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[1, Iplane]), (yinterp, zinterp), method='linear')
         Uz_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[2, Iplane]), (yinterp, zinterp), method='linear')
         ux=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Ux_i[i,j]
                if str(c)=='[nan]':
                   ux[i,j]=0
                else:
                   ux[i,j]=c[0]
         uy=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uy_i[i,j]
                if str(c)=='[nan]':
                   uy[i,j]=0
                else:
                   uy[i,j]=c[0]
         uz=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uz_i[i,j]
                if str(c)=='[nan]':
                   uz[i,j]=0
                else:
                   uz[i,j]=c[0]

         if sedfoam==0:
             original_shape = T.shape
             new_size = max(Iplane[0]) + 1  
             if original_shape[0] <= new_size:
                 T_expanded = np.zeros(new_size)
                 T_expanded[:original_shape[0]] = T
             else:
                 T_expanded = T
             T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')

         #Transversal streamwise velocity contourmap at x=xp/2
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(ux/Ub))
         contour = plt.contourf(yinterp/W, zinterp/H, ux/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar(contour)
         quiver_spacing = 8
         q = plt.quiver(yinterp[0,::quiver_spacing*2]/W, zinterp[:,0][::quiver_spacing//4]/H, uy[::quiver_spacing//4,::quiver_spacing*2]/Ub, uz[::quiver_spacing//4,::quiver_spacing*2]/Ub, width=0.002, scale=10, headwidth=2, color='black')
         if inst==0:
            cbar.set_label('$\\overline{u}/U_0$', fontsize=28)
         else:
            cbar.set_label('${u}/U_0$', fontsize=28)
         cbar.ax.tick_params(labelsize=24)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         plt.tick_params(axis='x', labelsize=24) 
         plt.tick_params(axis='y', labelsize=24)
         cbar.ax.yaxis.set_major_formatter(formatter)
         cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         plt.xlabel('$y/W$', fontsize=28)
         plt.ylabel('$z/H_0$', fontsize=28)
         Dxp=-0.5*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=0.5xp-streamwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")
         figcbar, axcbar = plt.subplots(figsize=(12, 1), dpi=500)
         vmin, vmax = -1.51, 1.51 
         cbar = figcbar.colorbar(plt.cm.ScalarMappable(cmap=vcolormap, norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=axcbar, orientation='horizontal')
         cbar.ax.tick_params(labelsize=24)
         ticks = np.round(np.linspace(vmin, vmax, 6), 2)
         cbar.set_ticks(ticks)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         if inst==0:
           cbar.set_label('$\\overline{u}/{U_0}$', fontsize=28)
         else:
           cbar.set_label('${u}/U_0$', fontsize=28)
         figcbar.savefig(sim + '-colorbar-trasversalplane-streamwise-velocity-field.pdf', bbox_inches="tight")
         plt.close(figcbar)
         
         #Transversal spanwise velocity contourmap at x=xp/2
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(uy/Ub))
         contour = plt.contourf(yinterp/H, zinterp/H, uy/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-W/2/H, color='black', linestyle='--', linewidth=2) 
         plt.axvline(x=+W/2/H, color='black', linestyle='--', linewidth=2) 
         cbar = plt.colorbar(contour)
         quiver_spacing = 8
         q = plt.quiver(yinterp[0,::quiver_spacing*2]/H, zinterp[:,0][::quiver_spacing//4]/H, uy[::quiver_spacing//4,::quiver_spacing*2]/Ub, uz[::quiver_spacing//4,::quiver_spacing*2]/Ub, width=0.002, scale=10, headwidth=2, color='white')
         if inst==0:
            cbar.set_label('$\\overline{v}/U_0$', fontsize=28)
         else:
            cbar.set_label('${v}/U_0$', fontsize=28)
         cbar.ax.tick_params(labelsize=24)
         plt.tick_params(axis='x', labelsize=24) 
         plt.tick_params(axis='y', labelsize=24)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         plt.xlabel('$y/H_0$', fontsize=28)
         plt.ylabel('$z/H_0$', fontsize=28)
         Dxp=-0.5*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         plt.xlim(-25,25)
         D05xp=Dxp
         if flagssurf==1 or flagasurf==1:
             cont = plt.contour(yinterp, zinterp, uy / Ub, levels=[0], colors='none')
             longest_path=max(cont.get_paths(),key=lambda path:len(path.vertices),default=None)
             if longest_path is not None:
              intersection_x=[]
              vertices=longest_path.vertices
              x_vals,y_vals=vertices[:,0],vertices[:,1]
              for i in range(len(y_vals)-1):
               if (y_vals[i]<=Dxp<=y_vals[i+1])or(y_vals[i+1]<=Dxp<=y_vals[i]):
                t=(Dxp-y_vals[i])/(y_vals[i+1]-y_vals[i])
                x_intersect=x_vals[i]+t*(x_vals[i+1]-x_vals[i])
                intersection_x.append(x_intersect)
              if intersection_x:
               Ltranspreadmax05xp=np.max(intersection_x)
               #plt.scatter(Ltranspreadmax05xp,Dxp,color='red',s=100,marker='o')
         filename='-transversalplane-x=0.5xp-spanwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Transversal vertical velocity contourmap at x=xp/2
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(uz))
         contour = plt.contourf(yinterp/W, zinterp/H, uz, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar(contour)
         if inst==0:
            cbar.set_label('$\\overline{U_z} (m/s)$', fontsize=12)
         else:
            cbar.set_label('${U_z} (m/s)$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxp=-0.5*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=0.5xp-vertical-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal velocity contourmap at x=xp !
      if flagssurf == 1:
         yi = np.linspace(2*yinterpmin_trans, 2*yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=xp-dx_trans,x<=xp+dx_trans))
         original_shape = U.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[1] <= new_size:
              U_expanded = np.zeros((original_shape[0], new_size))
              U_expanded[:, :original_shape[1]] = U
         else:
              U_expanded = U
         Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
         Uy_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[1, Iplane]), (yinterp, zinterp), method='linear')
         Uz_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[2, Iplane]), (yinterp, zinterp), method='linear')
         ux=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Ux_i[i,j]
                if str(c)=='[nan]':
                   ux[i,j]=0
                else:
                   ux[i,j]=c[0]
         uy=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uy_i[i,j]
                if str(c)=='[nan]':
                   uy[i,j]=0
                else:
                   uy[i,j]=c[0]
         uz=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uz_i[i,j]
                if str(c)=='[nan]':
                   uz[i,j]=0
                else:
                   uz[i,j]=c[0]

         if sedfoam==0:
             original_shape = T.shape
             new_size = max(Iplane[0]) + 1  
             if original_shape[0] <= new_size:
                 T_expanded = np.zeros(new_size)
                 T_expanded[:original_shape[0]] = T
             else:
                 T_expanded = T
             T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')

         #Transversal streamwise velocity contourmap at x=xp
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(ux/Ub))
         contour = plt.contourf(yinterp/W, zinterp/H, ux/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         #plt.contour(yinterp/W, zinterp/H, T_i/R, levels=[0.99], colors='white', linewidths=2)
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar(contour)
         quiver_spacing = 8
         q = plt.quiver(yinterp[0,::quiver_spacing*2]/W, zinterp[:,0][::quiver_spacing//4]/H, uy[::quiver_spacing//4,::quiver_spacing*2]/Ub, uz[::quiver_spacing//4,::quiver_spacing*2]/Ub, width=0.002, scale=10, headwidth=2, color='black')
         if inst==0:
            cbar.set_label('$\\overline{u}/U_0$', fontsize=28)
         else:
            cbar.set_label('${u}/U_0$', fontsize=28)
         cbar.ax.tick_params(labelsize=24)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         plt.tick_params(axis='x', labelsize=24) 
         plt.tick_params(axis='y', labelsize=24)
         cbar.ax.yaxis.set_major_formatter(formatter)
         cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         plt.xlabel('$y/W$', fontsize=28)
         plt.ylabel('$z/H_0$', fontsize=28)
         Dxp=-1.0*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=xp-streamwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")
         
         #Transversal spanwise velocity contourmap at x=xp
         fig = plt.figure(figsize=(32, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(uy/Ub))
         contour = plt.contourf(yinterp/H, zinterp/H, uy/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-W/2/H, color='black', linestyle='--', linewidth=2) 
         plt.axvline(x=+W/2/H, color='black', linestyle='--', linewidth=2) 
         cbar = plt.colorbar(contour)
         quiver_spacing = 8
         q = plt.quiver(yinterp[0,::quiver_spacing*2]/H, zinterp[:,0][::quiver_spacing//4]/H, uy[::quiver_spacing//4,::quiver_spacing*2]/Ub, uz[::quiver_spacing//4,::quiver_spacing*2]/Ub, width=0.002, scale=10, headwidth=2, color='white')
         if inst==0:
            cbar.set_label('$\\overline{v}/U_0$', fontsize=38)
            cbar.set_label('$\\overline{v} U_0^{-1}$', fontsize=38)
         else:
            cbar.set_label('${v} U_0^{-1}$', fontsize=38)
         cbar.ax.tick_params(labelsize=34)
         plt.tick_params(axis='x', labelsize=34) 
         plt.tick_params(axis='y', labelsize=34)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         plt.xlabel('$y/H_0$', fontsize=38)
         plt.ylabel('$z/H_0$', fontsize=38)
         plt.xlabel('$y H_0^{-1}$', fontsize=38)
         plt.ylabel('$z H_0^{-1}$', fontsize=38)
         Dxp=-1.0*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         plt.xlim(-25,25)
         plt.xlim(-50,50)
         D1xp=Dxp
         if flagssurf==1 or flagasurf==1:
             cont=plt.contour(yinterp,zinterp,uy/Ub,levels=[0],colors='none')
             longest_path=max(cont.get_paths(),key=lambda path:len(path.vertices),default=None)
             if longest_path is not None:
              intersection_x=[]
              vertices=longest_path.vertices
              x_vals,y_vals=vertices[:,0],vertices[:,1]
              for i in range(len(y_vals)-1):
               if (y_vals[i]<=Dxp<=y_vals[i+1])or(y_vals[i+1]<=Dxp<=y_vals[i]):
                t=(Dxp-y_vals[i])/(y_vals[i+1]-y_vals[i])
                x_intersect=x_vals[i]+t*(x_vals[i+1]-x_vals[i])
                intersection_x.append(x_intersect)
              if intersection_x:
               Ltranspreadmax1xp=np.max(intersection_x)
               #plt.scatter(Ltranspreadmax1xp/H,Dxp/H,color='red',s=100,marker='o')
         filename='-transversalplane-x=xp-spanwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Transversal vertical velocity contourmap at x=xp
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(uz))
         contour = plt.contourf(yinterp/W, zinterp/H, uz, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar(contour)
         if inst==0:
            cbar.set_label('$\\overline{U_z} (m/s)$', fontsize=12)
         else:
            cbar.set_label('${U_z} (m/s)$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxp=-1.0*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=xp-vertical-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal velocity contourmap at x=2xp !
      if flagssurf == 1:
         yi = np.linspace(2*yinterpmin_trans, 2*yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=2*xp-dx_trans,x<=2*xp+dx_trans))
         original_shape = U.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[1] <= new_size:
              U_expanded = np.zeros((original_shape[0], new_size))
              U_expanded[:, :original_shape[1]] = U
         else:
              U_expanded = U
         Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
         Uy_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[1, Iplane]), (yinterp, zinterp), method='linear')
         Uz_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[2, Iplane]), (yinterp, zinterp), method='linear')
         ux=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Ux_i[i,j]
                if str(c)=='[nan]':
                   ux[i,j]=0
                else:
                   ux[i,j]=c[0]
         uy=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uy_i[i,j]
                if str(c)=='[nan]':
                   uy[i,j]=0
                else:
                   uy[i,j]=c[0]
         uz=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uz_i[i,j]
                if str(c)=='[nan]':
                   uz[i,j]=0
                else:
                   uz[i,j]=c[0]

         if sedfoam==0:
             original_shape = T.shape
             new_size = max(Iplane[0]) + 1  
             if original_shape[0] <= new_size:
                 T_expanded = np.zeros(new_size)
                 T_expanded[:original_shape[0]] = T
             else:
                 T_expanded = T
             T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')

         #Transversal streamwise velocity contourmap at x=2xp
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(ux/Ub))
         contour = plt.contourf(yinterp/W, zinterp/H, ux/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar(contour)
         quiver_spacing = 8
         q = plt.quiver(yinterp[0,::quiver_spacing*2]/W, zinterp[:,0][::quiver_spacing//4]/H, uy[::quiver_spacing//4,::quiver_spacing*2]/Ub, uz[::quiver_spacing//4,::quiver_spacing*2]/Ub, width=0.002, scale=10, headwidth=2, color='black')
         if inst==0:
            cbar.set_label('$\\overline{u}/U_0$', fontsize=28)
         else:
            cbar.set_label('${u}/U_0$', fontsize=28)
         cbar.ax.tick_params(labelsize=24)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         plt.tick_params(axis='x', labelsize=24) 
         plt.tick_params(axis='y', labelsize=24)
         cbar.ax.yaxis.set_major_formatter(formatter)
         cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         plt.xlabel('$y/W$', fontsize=28)
         plt.ylabel('$z/H_0$', fontsize=28)
         Dxp=-2*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=2xp-streamwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")
         
         #Transversal spanwise velocity contourmap at x=2xp
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(uy/Ub))
         contour = plt.contourf(yinterp/H, zinterp/H, uy/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-W/2/H, color='black', linestyle='--', linewidth=2) 
         plt.axvline(x=+W/2/H, color='black', linestyle='--', linewidth=2) 
         cbar = plt.colorbar(contour)
         quiver_spacing = 8
         q = plt.quiver(yinterp[0,::quiver_spacing*2]/H, zinterp[:,0][::quiver_spacing//4]/H, uy[::quiver_spacing//4,::quiver_spacing*2]/Ub, uz[::quiver_spacing//4,::quiver_spacing*2]/Ub, width=0.002, scale=10, headwidth=2, color='white')
         if inst==0:
            cbar.set_label('$\\overline{v}/U_0$', fontsize=28)
         else:
            cbar.set_label('${v}/U_0$', fontsize=28)
         cbar.ax.tick_params(labelsize=24)
         plt.tick_params(axis='x', labelsize=24) 
         plt.tick_params(axis='y', labelsize=24)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         plt.xlabel('$y/H_0$', fontsize=28)
         plt.ylabel('$z/H_0$', fontsize=28)
         Dxp=-2*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         plt.xlim(-50,50)
         D2xp=Dxp
         if flagssurf==1 or flagasurf==1:
             cont=plt.contour(yinterp,zinterp,uy/Ub,levels=[0],colors='none')
             longest_path=max(cont.get_paths(),key=lambda path:len(path.vertices),default=None)
             if longest_path is not None:
              intersection_x=[]
              vertices=longest_path.vertices
              x_vals,y_vals=vertices[:,0],vertices[:,1]
              for i in range(len(y_vals)-1):
               if (y_vals[i]<=Dxp<=y_vals[i+1])or(y_vals[i+1]<=Dxp<=y_vals[i]):
                t=(Dxp-y_vals[i])/(y_vals[i+1]-y_vals[i])
                x_intersect=x_vals[i]+t*(x_vals[i+1]-x_vals[i])
                intersection_x.append(x_intersect)
              if intersection_x:
               Ltranspreadmax2xp=np.max(intersection_x)
               #plt.scatter(Ltranspreadmax2xp,Dxp,color='red',s=100,marker='o')
         filename='-transversalplane-x=2xp-spanwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Transversal vertical velocity contourmap at x=2xp
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(uz))
         contour = plt.contourf(yinterp/W, zinterp/H, uz, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar(contour)
         if inst==0:
            cbar.set_label('$\\overline{U_z} (m/s)$', fontsize=12)
         else:
            cbar.set_label('${U_z} (m/s)$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxp=-2*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=2xp-vertical-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal velocity contourmap at x=3xp !
      if flagssurf == 1:
         yi = np.linspace(3*yinterpmin_trans, 3*yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=3*xp-dx_trans,x<=3*xp+dx_trans))
         original_shape = U.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[1] <= new_size:
              U_expanded = np.zeros((original_shape[0], new_size))
              U_expanded[:, :original_shape[1]] = U
         else:
              U_expanded = U
         Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
         Uy_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[1, Iplane]), (yinterp, zinterp), method='linear')
         Uz_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[2, Iplane]), (yinterp, zinterp), method='linear')
         ux=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Ux_i[i,j]
                if str(c)=='[nan]':
                   ux[i,j]=0
                else:
                   ux[i,j]=c[0]
         uy=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uy_i[i,j]
                if str(c)=='[nan]':
                   uy[i,j]=0
                else:
                   uy[i,j]=c[0]
         uz=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uz_i[i,j]
                if str(c)=='[nan]':
                   uz[i,j]=0
                else:
                   uz[i,j]=c[0]

         if sedfoam==0:
             original_shape = T.shape
             new_size = max(Iplane[0]) + 1  
             if original_shape[0] <= new_size:
                 T_expanded = np.zeros(new_size)
                 T_expanded[:original_shape[0]] = T
             else:
                 T_expanded = T
             T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')

         #Transversal streamwise velocity contourmap at x=3xp
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(ux/Ub))
         contour = plt.contourf(yinterp/W, zinterp/H, ux/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         #plt.contour(yinterp/W, zinterp/H, T_i/R, levels=[0.99], colors='white', linewidths=2)
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar(contour)
         quiver_spacing = 8
         q = plt.quiver(yinterp[0,::quiver_spacing*2]/W, zinterp[:,0][::quiver_spacing//4]/H, uy[::quiver_spacing//4,::quiver_spacing*2]/Ub, uz[::quiver_spacing//4,::quiver_spacing*2]/Ub, width=0.002, scale=10, headwidth=2, color='black')
         if inst==0:
            cbar.set_label('$\\overline{u}/U_0$', fontsize=28)
         else:
            cbar.set_label('${u}/U_0$', fontsize=28)
         cbar.ax.tick_params(labelsize=24)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         plt.tick_params(axis='x', labelsize=24) 
         plt.tick_params(axis='y', labelsize=24)
         cbar.ax.yaxis.set_major_formatter(formatter)
         cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         plt.xlabel('$y/W$', fontsize=28)
         plt.ylabel('$z/H_0$', fontsize=28)
         Dxp=-3*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=3xp-streamwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")
         
         #Transversal spanwise velocity contourmap at x=3xp
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(uy/Ub))
         contour = plt.contourf(yinterp/H, zinterp/H, uy/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-W/2/H, color='black', linestyle='--', linewidth=2) 
         plt.axvline(x=+W/2/H, color='black', linestyle='--', linewidth=2) 
         cbar = plt.colorbar(contour)
         quiver_spacing = 8
         q = plt.quiver(yinterp[0,::quiver_spacing*2]/H, zinterp[:,0][::quiver_spacing//4]/H, uy[::quiver_spacing//4,::quiver_spacing*2]/Ub, uz[::quiver_spacing//4,::quiver_spacing*2]/Ub, width=0.002, scale=10, headwidth=2, color='white')
         if inst==0:
            cbar.set_label('$\\overline{v}/U_0$', fontsize=28)
         else:
            cbar.set_label('${v}/U_0$', fontsize=28)
         cbar.ax.tick_params(labelsize=24)
         plt.tick_params(axis='x', labelsize=24) 
         plt.tick_params(axis='y', labelsize=24)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
         cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         plt.xlabel('$y/H_0$', fontsize=28)
         plt.ylabel('$z/H_0$', fontsize=28)
         Dxp=-3*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         plt.xlim(-75,75)
         D3xp=Dxp
         if flagssurf==1 or flagasurf==1:
             cont=plt.contour(yinterp,zinterp,uy/Ub,levels=[0],colors='none')
             longest_path=max(cont.get_paths(),key=lambda path:len(path.vertices),default=None)
             if longest_path is not None:
              intersection_x=[]
              vertices=longest_path.vertices
              x_vals,y_vals=vertices[:,0],vertices[:,1]
              for i in range(len(y_vals)-1):
               if (y_vals[i]<=Dxp<=y_vals[i+1])or(y_vals[i+1]<=Dxp<=y_vals[i]):
                t=(Dxp-y_vals[i])/(y_vals[i+1]-y_vals[i])
                x_intersect=x_vals[i]+t*(x_vals[i+1]-x_vals[i])
                intersection_x.append(x_intersect)
              if intersection_x:
               Ltranspreadmax3xp=np.max(intersection_x)
               #plt.scatter(Ltranspreadmax3xp,Dxp,color='red',s=100,marker='o')
         filename='-transversalplane-x=3xp-spanwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Transversal vertical velocity contourmap at x=3xp
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'         
         abs_max = np.nanmax(np.abs(uz))
         contour = plt.contourf(yinterp/W, zinterp/H, uz, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar(contour)
         if inst==0:
            cbar.set_label('$\\overline{U_z} (m/s)$', fontsize=12)
         else:
            cbar.set_label('${U_z} (m/s)$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxp=-3*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=3xp-vertical-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         if flagssurf==1 or flagasurf==1:
             print(f"######################################################################################################################################################################################################")
             print(f"Lateral plume spread based on transversal maps of spanwise velocity")
             print(f"Maximum lateral spread at x=xp/2: Lspread_max_05xp={Ltranspreadmax05xp:.5f} m")
             print(f"Maximum lateral spread over channel width at x=xp/2: Lspread_max_05xp/W={Ltranspreadmax05xp/W:.5f}")
             print(f"Depth at x=xp/2: Lspread_max_05xp={D05xp:.5f} m")
             print(f"Depth at x=xp/2 over channel height: Lspread_max_05xp={D05xp/H:.5f}")
             print(f"Maximum lateral spread at x=xp: Lspread_max_1xp={Ltranspreadmax1xp:.5f} m")
             print(f"Maximum lateral spread over channel width at x=xp: Lspread_max_1xp/W={Ltranspreadmax1xp/W:.5f}")
             print(f"Depth at x=xp: Lspread_max_1xp={D1xp:.5f} m")
             print(f"Depth at x=xp over channel height: Lspread_max_1xp={D1xp/H:.5f}")
             print(f"Maximum lateral spread at x=2xp: Lspread_max_2xp={Ltranspreadmax2xp:.5f} m")
             print(f"Maximum lateral spread over channel width at x=2xp: Lspread_max_2xp/W={Ltranspreadmax2xp/W:.5f}")
             print(f"Depth at x=2xp: Lspread_max_2xp={D2xp:.5f} m")
             print(f"Depth at x=2xp over channel height: Lspread_max_2xp={D2xp/H:.5f}")
             print(f"Maximum lateral spread at x=3xp: Lspread_max_3xp={Ltranspreadmax3xp:.5f} m")
             print(f"Maximum lateral spread over channel width at x=3xp: Lspread_max_3xp/W={Ltranspreadmax3xp/W:.5f}")
             print(f"Depth at x=3xp: Lspread_max_3xp={D3xp:.5f} m")
             print(f"Depth at x=3xp over channel height: Lspread_max_3xp={D3xp/H:.5f}")
             print(f"######################################################################################################################################################################################################")

         #Lateral plume spread based on transvere plane maps 
         if flagssurf==1 and sedfoam==0:
            fig=plt.figure(figsize=(6,4),dpi=500)
            plt.rcParams.update({'font.size':11,'font.family':'serif','font.serif':['Computer Modern Roman'],'text.usetex':True})
            plt.rcParams['xtick.direction']='in'
            plt.rcParams['ytick.direction']='in'
            xtransversalplane=np.array([0.5*xp,xp,2*xp,3*xp])
            Ltranspread=np.array([Ltranspreadmax05xp/W,Ltranspreadmax1xp/W,Ltranspreadmax2xp/W,Ltranspreadmax3xp/W])
            csvname='-x-based-on-transverse-maps.csv'
            csvname=sim+csvname
            pd.DataFrame({'x':xtransversalplane}).to_csv(csvname,index=False)
            csvname='-Ltranspread-based-on-transverse-maps.csv'
            csvname=sim+csvname
            pd.DataFrame({'Lspread':Ltranspread}).to_csv(csvname,index=False)
            plt.plot(xtransversalplane/xp,Ltranspread, linestyle=' ',marker='s',markersize=5, markerfacecolor='none', color='black')
            plt.axhline(y=0,color='black',linestyle='--',linewidth=1)
            plt.axvline(x=1,color='black',linestyle='--',linewidth=1)
            plt.xlabel('$x/x_p$',fontsize=12)
            plt.ylabel('$\\overline{L_{s}}/W$' if inst==0 else '${L_{s}}/W$',fontsize=12)
            filename='-Ltranspread-based-on-transverse-maps.pdf'
            savename=sim+filename
            plt.savefig(savename,bbox_inches='tight')
            plt.close()
            print(f'Plotted "{savename}" successfully!')
         
         #Plume depth based on transvere plane maps
         if flagssurf==1 and sedfoam==0:
            fig=plt.figure(figsize=(6,4),dpi=500)
            plt.rcParams.update({'font.size':11,'font.family':'serif','font.serif':['Computer Modern Roman'],'text.usetex':True})
            plt.rcParams['xtick.direction']='in'
            plt.rcParams['ytick.direction']='in'
            xtransversalplane=np.array([0.5*xp,xp,2*xp,3*xp])
            Dtransversalplane=np.abs([D05xp/H,D1xp/H,D2xp/H,D3xp/H])
            csvname='-x-based-on-transverse-maps.csv'
            csvname=sim+csvname
            pd.DataFrame({'x':xtransversalplane}).to_csv(csvname,index=False)
            csvname='-Dtransversalplane-based-on-transverse-maps.csv'
            csvname=sim+csvname
            pd.DataFrame({'Dtransversalplane':Dtransversalplane}).to_csv(csvname,index=False)
            plt.plot(xtransversalplane/xp, Dtransversalplane, linestyle=' ',marker='s',markersize=5, markerfacecolor='none', color='black')
            plt.axhline(y=1,color='black',linestyle='--',linewidth=1)
            plt.axvline(x=1,color='black',linestyle='--',linewidth=1)
            plt.xlabel('$x/x_p$',fontsize=12)
            plt.ylabel('$\\overline{D}/H$' if inst==0 else '$D/H$',fontsize=12)
            filename='-Dtransversalplane-based-on-transverse-maps.pdf'
            savename=sim+filename
            plt.savefig(savename,bbox_inches='tight')
            plt.close()
            print(f'Plotted "{savename}" successfully!')
    
   #Transversalplane streamlines map !
   if flagstreamtrans == 1:
      print(f"############################################################################################")
      print(f"Transversalplane streamlines contourmap !")
      print(f"############################################################################################")
      #Transversalplane velocity contourmap !
      yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
      zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
      yinterp, zinterp = np.meshgrid(yi, zi)
      Iplane=np.where(np.logical_and(x>=Xplane_trans-dx_trans,x<=Xplane_trans+dx_trans))
      original_shape = U.shape
      new_size = max(Iplane[0]) + 1  
      if original_shape[1] <= new_size:
         U_expanded = np.zeros((original_shape[0], new_size))
         U_expanded[:, :original_shape[1]] = U
      else:
         U_expanded = U
      Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
      Uy_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[1, Iplane]), (yinterp, zinterp), method='linear')
      Uz_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[2, Iplane]), (yinterp, zinterp), method='linear')
      ux=np.zeros((ngridz_trans,ngridy_trans))
      for i in range (0,ngridz_trans):
        for j in range (0,ngridy_trans):
            c=Ux_i[i,j]
            if str(c)=='[nan]':
               ux[i,j]=0
            else:
               ux[i,j]=c[0]
      uy=np.zeros((ngridz_trans,ngridy_trans))
      for i in range (0,ngridz_trans):
        for j in range (0,ngridy_trans):
            c=Uy_i[i,j]
            if str(c)=='[nan]':
               uy[i,j]=0
            else:
               uy[i,j]=c[0]
      uz=np.zeros((ngridz_trans,ngridy_trans))
      for i in range (0,ngridz_trans):
        for j in range (0,ngridy_trans):
            c=Uz_i[i,j]
            if str(c)=='[nan]':
               uz[i,j]=0
            else:
               uz[i,j]=c[0]
      
      #Transversalplane streamlines contourmap at x=0 spanwise velocity coloured !
      Dxp=-0*tan(beta/180*pi)-H;
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      abs_max = np.max(np.abs(uy))
      contour = plt.contourf(yinterp/W, zinterp/H, uy, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
      speed = np.sqrt(np.nan_to_num(uy, nan=0.0, posinf=0.0, neginf=0.0)**2 + np.nan_to_num(0, nan=0.0, posinf=0.0, neginf=0.0)**2)
      plt.streamplot(yinterp/W, zinterp/H, uy, uz, color='black', linewidth=2*speed/speed.max(), density=(2.0, 2.0))
      cbar = plt.colorbar(contour)
      if inst==0:
        cbar.set_label('$\\overline{U_y} (m/s)$', fontsize=12)
      else:
        cbar.set_label('${U_y} (m/s)$', fontsize=12)
      cbar.ax.tick_params(labelsize=10)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.xlabel('$y/W$', fontsize=12)
      plt.ylabel('$z/H_0$', fontsize=12)
      plt.ylim(Dxp/H,0)
      filename='-transverseplane-x=0-streamlines-spanwise-velocity-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Transversalplane streamlines contourmap at x=0 vertical velocity coloured !
      Dxp=-0*tan(beta/180*pi)-H;
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      abs_max = np.max(np.abs(uz))
      contour = plt.contourf(yinterp/W, zinterp/H, uz, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
      speed = np.sqrt(np.nan_to_num(0, nan=0.0, posinf=0.0, neginf=0.0)**2 + np.nan_to_num(uz, nan=0.0, posinf=0.0, neginf=0.0)**2)
      plt.streamplot(yinterp/W, zinterp/H, uy, uz, color='black', linewidth=2*speed/speed.max(), density=(2.0, 2.0))
      cbar = plt.colorbar(contour)
      if inst==0:
        cbar.set_label('$\\overline{U_z} (m/s)$', fontsize=12)
      else:
        cbar.set_label('${U_z} (m/s)$', fontsize=12)
      cbar.ax.tick_params(labelsize=10)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.xlabel('$y/W$', fontsize=12)
      plt.ylabel('$z/H_0$', fontsize=12)
      plt.ylim(Dxp/H,0)
      filename='-transverseplane-x=0-streamlines-vertical-velocity-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Transversal streamlines contourmap at x=xp/2 !
      if flagssurf == 1:
         yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=0.5*xp-dx_trans,x<=0.5*xp+dx_trans))
         original_shape = U.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[1] <= new_size:
              U_expanded = np.zeros((original_shape[0], new_size))
              U_expanded[:, :original_shape[1]] = U
         else:
              U_expanded = U
         Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
         Uy_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[1, Iplane]), (yinterp, zinterp), method='linear')
         Uz_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[2, Iplane]), (yinterp, zinterp), method='linear')
         ux=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Ux_i[i,j]
                if str(c)=='[nan]':
                   ux[i,j]=0
                else:
                   ux[i,j]=c[0]
         uy=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uy_i[i,j]
                if str(c)=='[nan]':
                   uy[i,j]=0
                else:
                   uy[i,j]=c[0]
         uz=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uz_i[i,j]
                if str(c)=='[nan]':
                   uz[i,j]=0
                else:
                   uz[i,j]=c[0]

         #Transversalplane streamlines contourmap at x=xp/2 spanwise velocity coloured !
         Dxp=-0.5*xp*tan(beta/180*pi)-H;
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         abs_max = np.max(np.abs(uy))
         contour = plt.contourf(yinterp/W, zinterp/H, uy, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
         speed = np.sqrt(np.nan_to_num(uy, nan=0.0, posinf=0.0, neginf=0.0)**2 + np.nan_to_num(0, nan=0.0, posinf=0.0, neginf=0.0)**2)
         plt.streamplot(yinterp/W, zinterp/H, uy, uz, color='black', linewidth=2*speed/speed.max(), density=(2.0, 2.0))
         cbar = plt.colorbar(contour)
         if inst==0:
           cbar.set_label('$\\overline{U_y} (m/s)$', fontsize=12)
         else:
           cbar.set_label('${U_y} (m/s)$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         plt.ylim(Dxp,0)
         filename='-transverseplane-x=0.5xp-streamlines-spanwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")
         
         #Transversalplane streamlines contourmap at x=xp/2 vertical velocity coloured !
         Dxp=-0.5*xp*tan(beta/180*pi)-H;
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         abs_max = np.max(np.abs(uz))
         contour = plt.contourf(yinterp/W, zinterp/H, uz, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
         speed = np.sqrt(np.nan_to_num(0, nan=0.0, posinf=0.0, neginf=0.0)**2 + np.nan_to_num(uz, nan=0.0, posinf=0.0, neginf=0.0)**2)
         plt.streamplot(yinterp/W, zinterp/H, uy, uz, color='black', linewidth=2*speed/speed.max(), density=(2.0, 2.0))
         cbar = plt.colorbar(contour)
         if inst==0:
           cbar.set_label('$\\overline{U_z} (m/s)$', fontsize=12)
         else:
           cbar.set_label('${U_z} (m/s)$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         plt.ylim(Dxp/H,0)
         filename='-transverseplane-x=0.5xp-streamlines-vertical-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal streamlines contourmap at x=xp !
      if flagssurf == 1:
         yi = np.linspace(yinterpmin_trans, yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=xp-dx_trans,x<=xp+dx_trans))
         original_shape = U.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[1] <= new_size:
              U_expanded = np.zeros((original_shape[0], new_size))
              U_expanded[:, :original_shape[1]] = U
         else:
              U_expanded = U
         Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
         Uy_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[1, Iplane]), (yinterp, zinterp), method='linear')
         Uz_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[2, Iplane]), (yinterp, zinterp), method='linear')
         ux=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Ux_i[i,j]
                if str(c)=='[nan]':
                   ux[i,j]=0
                else:
                   ux[i,j]=c[0]
         uy=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uy_i[i,j]
                if str(c)=='[nan]':
                   uy[i,j]=0
                else:
                   uy[i,j]=c[0]
         uz=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uz_i[i,j]
                if str(c)=='[nan]':
                   uz[i,j]=0
                else:
                   uz[i,j]=c[0]

         #Transversalplane streamlines contourmap at x=xp spanwise velocity coloured !
         Dxp=-xp*tan(beta/180*pi)-H;
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         abs_max = np.max(np.abs(uy))
         contour = plt.contourf(yinterp/W, zinterp/H, uy, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
         speed = np.sqrt(np.nan_to_num(uy, nan=0.0, posinf=0.0, neginf=0.0)**2 + np.nan_to_num(0, nan=0.0, posinf=0.0, neginf=0.0)**2)
         plt.streamplot(yinterp/W, zinterp/H, uy, uz, color='black', linewidth=2*speed/speed.max(), density=(2.0, 2.0))
         cbar = plt.colorbar(contour)
         if inst==0:
           cbar.set_label('$\\overline{U_y} (m/s)$', fontsize=12)
         else:
           cbar.set_label('${U_y} (m/s)$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         plt.ylim(Dxp/W,0)
         filename='-transverseplane-x=xp-streamlines-spanwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Transversalplane streamlines contourmap at x=xp vertical velocity coloured !
         Dxp=-xp*tan(beta/180*pi)-H;
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         abs_max = np.max(np.abs(uz))
         contour = plt.contourf(yinterp/W, zinterp/H, uz, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
         speed = np.sqrt(np.nan_to_num(0, nan=0.0, posinf=0.0, neginf=0.0)**2 + np.nan_to_num(uz, nan=0.0, posinf=0.0, neginf=0.0)**2)
         plt.streamplot(yinterp/W, zinterp/H, uy, uz, color='black', linewidth=2*speed/speed.max(), density=(2.0, 2.0))
         cbar = plt.colorbar(contour)
         if inst==0:
           cbar.set_label('$\\overline{U_z} (m/s)$', fontsize=12)
         else:
           cbar.set_label('${U_z} (m/s)$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         plt.ylim(Dxp/H,0)
         filename='-transverseplane-x=xp-streamlines-vertical-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

      #Transversal streamlines contourmap at x=2*xp !
      if flagssurf == 1:
         yi = np.linspace(2*yinterpmin_trans, 2*yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=2*xp-dx_trans,x<=2*xp+dx_trans))
         original_shape = U.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[1] <= new_size:
              U_expanded = np.zeros((original_shape[0], new_size))
              U_expanded[:, :original_shape[1]] = U
         else:
              U_expanded = U
         Ux_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[0, Iplane]), (yinterp, zinterp), method='linear')
         Uy_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[1, Iplane]), (yinterp, zinterp), method='linear')
         Uz_i = griddata((y[Iplane], z[Iplane]), np.transpose(U_expanded[2, Iplane]), (yinterp, zinterp), method='linear')
         ux=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Ux_i[i,j]
                if str(c)=='[nan]':
                   ux[i,j]=0
                else:
                   ux[i,j]=c[0]
         uy=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uy_i[i,j]
                if str(c)=='[nan]':
                   uy[i,j]=0
                else:
                   uy[i,j]=c[0]
         uz=np.zeros((ngridz_trans,ngridy_trans))
         for i in range (0,ngridz_trans):
            for j in range (0,ngridy_trans):
                c=Uz_i[i,j]
                if str(c)=='[nan]':
                   uz[i,j]=0
                else:
                   uz[i,j]=c[0]

         #Transversalplane streamlines contourmap at x=2*xp spanwise velocity coloured !
         Dxp=-2*xp*tan(beta/180*pi)-H;
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         abs_max = np.max(np.abs(uy))
         contour = plt.contourf(yinterp/W, zinterp/H, uy, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
         speed = np.sqrt(np.nan_to_num(uy, nan=0.0, posinf=0.0, neginf=0.0)**2 + np.nan_to_num(0, nan=0.0, posinf=0.0, neginf=0.0)**2)
         plt.streamplot(yinterp/W, zinterp/H, uy, uz, color='black', linewidth=2*speed/speed.max(), density=(2.0, 2.0))
         cbar = plt.colorbar(contour)
         if inst==0:
           cbar.set_label('$\\overline{U_y} (m/s)$', fontsize=12)
         else:
           cbar.set_label('${U_y} (m/s)$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H$', fontsize=12)
         plt.ylim(Dxp/H,0)
         filename='-transverseplane-x=2xp-streamlines-spanwise-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Transversalplane streamlines contourmap at x=2*xp vertical velocity coloured !
         Dxp=-2*xp*tan(beta/180*pi)-H;
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         abs_max = np.max(np.abs(uz))
         contour = plt.contourf(yinterp/W, zinterp/H, uz, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
         speed = np.sqrt(np.nan_to_num(0, nan=0.0, posinf=0.0, neginf=0.0)**2 + np.nan_to_num(uz, nan=0.0, posinf=0.0, neginf=0.0)**2)
         plt.streamplot(yinterp/W, zinterp/H, uy, uz, color='black', linewidth=2*speed/speed.max(), density=(2.0, 2.0))
         cbar = plt.colorbar(contour)
         if inst==0:
           cbar.set_label('$\\overline{U_z} (m/s)$', fontsize=12)
         else:
           cbar.set_label('${U_z} (m/s)$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         plt.ylim(Dxp/H,0)
         filename='-transverseplane-x=2xp-streamlines-vertical-velocity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

   #Transversalplane computed eddy viscosity map !
   if flagcompeddvisctrans == 1 and inst != 1 and sedfoam == 0:
      print(f"############################################################################################")
      print(f"Transversalplane computed eddy viscosity contourmap !")
      print(f"############################################################################################")
      #Transversal computed eddy viscosity contourmap at x=xp !
      if flagssurf == 1:
         yi = np.linspace(2*yinterpmin_trans, 2*yinterpmax_trans, ngridy_trans)
         zi = np.linspace(zinterpmin_trans, zinterpmax_trans, ngridz_trans)
         yinterp, zinterp = np.meshgrid(yi, zi)
         Iplane=np.where(np.logical_and(x>=xp-dx_trans,x<=xp+dx_trans))
         original_shape = T.shape
         new_size = max(Iplane[0]) + 1  
         if original_shape[0] <= new_size:
            T_expanded = np.zeros(new_size)
            T_expanded[:original_shape[0]] = T
         else:
            T_expanded = T
         T_i = griddata((y[Iplane], z[Iplane]), T_expanded[Iplane], (yinterp, zinterp), method='linear')
         Umxy_i = griddata((y[Iplane], z[Iplane]), Umxy[Iplane], (yinterp, zinterp), method='linear')
         Rxy_i = griddata((y[Iplane], z[Iplane]), Rxy[Iplane], (yinterp, zinterp), method='linear')
         Ryz_i = griddata((y[Iplane], z[Iplane]), Ryz[Iplane], (yinterp, zinterp), method='linear')
      
         rxy = np.zeros((ngridz_trans,ngridy_trans))
         for i in range (ngridz_trans):
           for j in range (ngridy_trans):
               c = Rxy_i[i,j]
               if np.isnan(c):
                  rxy[i,j]=0
               else:
                  rxy[i,j]=c.item()

         ryz = np.zeros((ngridz_trans,ngridy_trans))
         for i in range (ngridz_trans):
           for j in range (ngridy_trans):
               c=Ryz_i[i,j]
               if np.isnan(c):
                  ryz[i,j]=0
               else:
                  ryz[i,j]=c.item()

         umxy = np.zeros((ngridz_trans,ngridy_trans))
         for i in range (ngridz_trans):
           for j in range (ngridy_trans):
               c=Umxy_i[i,j]
               if np.isnan(c):
                  umxy[i,j]=0
               else:
                  umxy[i,j]=c.item()
         
         #Transversal mean salinity contourmap !
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'           
         plt.contourf(yinterp/W, zinterp/H, T_i/(R), cmap=scolormap, levels=np.linspace(0, np.nanmax(T_i/R), 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         if inst==0:
            cbar.set_label('$\\overline{R}$/${R}_{0}$', fontsize=12)
         else:
            cbar.set_label('${R}$/${R}_{0}$', fontsize=12)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.xlabel('$y/W$', fontsize=12)
         plt.ylabel('$z/H_0$', fontsize=12)
         Dxp=-xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=xp-salinity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Transversal mean velocity gradient contourmap !
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'      
         abs_max = np.nanmax(np.abs(umxy))
         plt.contourf(yinterp/W, zinterp/H, np.abs(umxy), cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         cbar.set_label(r'$|\overline{\mathrm{d}U/\mathrm{d}y}| 1/s$', fontsize=28)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.tick_params(axis='x', labelsize=24) 
         plt.tick_params(axis='y', labelsize=24)
         plt.xlabel('$y/W$', fontsize=28)
         plt.ylabel('$z/H_0$', fontsize=28)
         Dxp=-1.0*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=xp-mean-velocity-gradient-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Transversal reynolds stress contourmap !
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in' 
         abs_max = np.nanmax(np.abs(ryz))
         plt.contourf(yinterp/W, zinterp/H, np.abs(ryz), cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar = plt.colorbar()
         cbar.set_label(r'$|-\overline{v^\prime w^\prime}| m^2/s^2$', fontsize=28)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.tick_params(axis='x', labelsize=24) 
         plt.tick_params(axis='y', labelsize=24)
         plt.xlabel('$y/W$', fontsize=28)
         plt.ylabel('$z/H_0$', fontsize=28)
         Dxp=-1.0*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=xp-reynolds-stess-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

         #Transversal computed eddy viscosity contourmap at x=xp !
         fig = plt.figure(figsize=(16, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'      
         mask = (np.abs(umxy) > 1e-2) & (np.abs(rxy) > 1e-5)
         edvi = np.full_like(umxy, np.nan)
         edvi[mask] = 1 / (np.abs(umxy[mask]) / np.abs(rxy[mask]))
         #vmax, vmin = np.nanmax(edvi), np.nanmin(edvi)
         #plt.contourf(yinterp/W, zinterp/H, edvi, cmap=vcolormap, levels=100, vmin=vmin, vmax=vmax)
         plt.contourf(yinterp/W, zinterp/H, edvi, cmap=ecolormap, levels=np.linspace(0, 0.01, 100))
         cbar = plt.colorbar()
         plt.axvline(x=-1/2, color='black', linestyle='--', linewidth=1) 
         plt.axvline(x=+1/2, color='black', linestyle='--', linewidth=1) 
         cbar.set_label(r'$|\frac{-\overline{u^{\prime}v^{\prime}}}{\overline{\mathrm{d}U/\mathrm{d}y}}| m^2/s$', fontsize=28)
         cbar.ax.tick_params(labelsize=10)
         formatter = ScalarFormatter(useMathText=True)
         formatter.set_powerlimits((0, 0)) 
         cbar.ax.yaxis.set_major_formatter(formatter)
         plt.tick_params(axis='x', labelsize=24) 
         plt.tick_params(axis='y', labelsize=24)
         plt.xlabel('$y/W$', fontsize=28)
         plt.ylabel('$z/H_0$', fontsize=28)
         Dxp=-1.0*xp*tan(beta/180*pi)-H;
         plt.ylim(Dxp/H,0)
         filename='-transversalplane-x=xp-computed-eddy-viscosity-field.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")

   #Inclined plane salinity map !
   if flagsclin == 1 and sedfoam==0:
      print(f"############################################################################################")
      print(f"Inclined plane salinity contourmap !")
      print(f"############################################################################################")
      #Inclined plane salinity contourmap !
      xcl=x*cos(beta/180*pi)-z*sin(beta/180*pi)
      ycl=y
      zcl=x*sin(beta/180*pi)+z*cos(beta/180*pi)+H*cos(beta/180*pi)
      xcli = np.linspace(xinterpmin_clin, xinterpmax_clin, ngridx_clin)
      ycli = np.linspace(yinterpmin_clin, yinterpmax_clin, ngridy_clin)
      xclinterp, yclinterp = np.meshgrid(xcli, ycli)
      Iplane=np.where(np.logical_and(zcl>=Zvert_clin-dz_clin,zcl<=Zvert_clin+dz_clin))
      T_i = griddata((xcl[Iplane], ycl[Iplane]), T[Iplane], (xclinterp, yclinterp), method='linear')

      T=np.zeros((ngridy_clin,ngridx_clin))
      for i in range (0,ngridy_clin):
          for j in range (0,ngridx_clin):
              c=T_i[i,j]
              if np.isnan(c):
                 T[i,j]=0
              else:
                 T[i,j]=c.item()

      fig = plt.figure(figsize=(6, 4),dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      plt.contourf(xclinterp/W, yclinterp/W, T/R, cmap=scolormap, levels=np.linspace(0, np.max(T/R), 100))
      print("NaN values in T/R:", np.isnan(T/R).any())
      print("Inf values in T/R:", np.isinf(T/R).any())
      cbar = plt.colorbar()
      if flagssurf == 1 and sedfoam==0:
         plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
         plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
      plt.contour(xclinterp/W, yclinterp/W, T/R, levels=[0.99], colors='black', linewidths=2)
      if inst==0:
         cbar.set_label('$\\overline{R}$/${R}_{0}$', fontsize=12)
      else:
         cbar.set_label('${R}$/${R}_{0}$', fontsize=12)
      cbar.ax.tick_params(labelsize=10)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.xlabel('$x/W$', fontsize=12)
      plt.ylabel('$y/W$', fontsize=12)
      filename='-inclineplane-salinity-field.pdf'
      savename = sim + filename
      plt.savefig(savename)
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

   #Inclined plane sediment concentration map !
   if flagaclin == 1 and sedfoam==1:
      print(f"############################################################################################")
      print(f"Inclined plane sediment concentration contourmap !")
      print(f"############################################################################################")
      #Inclined plane salinity contourmap !
      xcl=x*cos(beta/180*pi)-z*sin(beta/180*pi)
      ycl=y
      zcl=x*sin(beta/180*pi)+z*cos(beta/180*pi)+H*cos(beta/180*pi)
      xcli = np.linspace(xinterpmin_clin, xinterpmax_clin, ngridx_clin)
      ycli = np.linspace(yinterpmin_clin, yinterpmax_clin, ngridy_clin)
      xclinterp, yclinterp = np.meshgrid(xcli, ycli)
      Iplane=np.where(np.logical_and(zcl>=Zvert_clin-dz_clin,zcl<=Zvert_clin+dz_clin))
      a_i = griddata((xcl[Iplane], ycl[Iplane]), alpha[Iplane], (xclinterp, yclinterp), method='linear')

      a=np.zeros((ngridy_clin,ngridx_clin))
      for i in range (0,ngridy_clin):
          for j in range (0,ngridx_clin):
              c=a_i[i,j]
              if np.isnan(c):
                 a[i,j]=0
              else:
                 a[i,j]=c.item()

      fig = plt.figure(figsize=(6, 4),dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      plt.contourf(xclinterp/W, yclinterp/W, a/a0, cmap=scolormap, levels=np.linspace(0, 1, 100))
      print("NaN values in a:", np.isnan(a).any())
      print("Inf values in a:", np.isinf(a).any())
      cbar = plt.colorbar()
      if flagasurf == 1 and sedfoam==1:
         plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
         plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
      #plt.contour(xclinterp/W, yclinterp/W, a/10, levels=[0.99], colors='black', linewidths=2)
      if inst==0:
         cbar.set_label('$\\overline{\\alpha}/a_0$', fontsize=28)
      else:
         cbar.set_label('${\\alpha}/a_0$', fontsize=28)
      cbar.ax.tick_params(labelsize=24)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.tick_params(axis='x', labelsize=24)  
      plt.tick_params(axis='y', labelsize=24)
      plt.xlabel('$x/W$', fontsize=28)
      plt.ylabel('$y/W$', fontsize=28)
      filename='-inclineplane-sediment-concentration-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")         

   #Inclined plane velocity map !
   if flaguclin == 1:
      print(f"############################################################################################")
      print(f"Inclined plane velocity contourmap !")
      print(f"############################################################################################")
      #Inclined plane velocity contourmap !
      xcl=x*cos(beta/180*pi)-z*sin(beta/180*pi)
      ycl=y
      zcl=x*sin(beta/180*pi)+z*cos(beta/180*pi)+H*cos(beta/180*pi)
      yinterpmin_clin=2*yinterpmin_clin
      yinterpmax_clin=2*yinterpmax_clin
      xinterpmin_clin=xinterpmin_clin
      xinterpmax_clin=xinterpmax_clin
      xcli = np.linspace(xinterpmin_clin, xinterpmax_clin, ngridx_clin)
      ycli = np.linspace(yinterpmin_clin, yinterpmax_clin, ngridy_clin)
      xclinterp, yclinterp = np.meshgrid(xcli, ycli)
      Iplane=np.where(np.logical_and(zcl>=Zvert_clin-dz_clin,zcl<=Zvert_clin+dz_clin))
      Ux_i = griddata((xcl[Iplane], ycl[Iplane]), np.transpose(U[0, Iplane]), (xclinterp, yclinterp), method='linear')
      Uy_i = griddata((xcl[Iplane], ycl[Iplane]), np.transpose(U[1, Iplane]), (xclinterp, yclinterp), method='linear') 
      Uz_i = griddata((xcl[Iplane], ycl[Iplane]), np.transpose(U[2, Iplane]), (xclinterp, yclinterp), method='linear') 

      ux=np.zeros((ngridy_clin,ngridx_clin))
      uy=np.zeros((ngridy_clin,ngridx_clin))
      uz=np.zeros((ngridy_clin,ngridx_clin))
      for i in range (0,ngridy_clin):
       for j in range (0,ngridx_clin):
         c=Ux_i[i,j]
         if str(c) == '[nan]':
                ux[i,j] = 0
         else:
                ux[i,j] = c[0]
         d=Uy_i[i,j]
         if str(d) == '[nan]':
                uy[i,j] = 0
         else:
                uy[i,j] = d[0]
         e=Uz_i[i,j]
         if str(e) == '[nan]':
                uz[i,j] = 0
         else:
                uz[i,j] = e[0]
     
      ut = (ux**2 + uz**2)**0.5
      epsilon = 1e-12 
      gamma = np.arctan(uz / (ux + epsilon)) 
      alphagon = beta/180*pi - gamma 
      us = np.cos(alphagon)*ut
      un = np.sin(alphagon)*ut
      us[ux < 0] = -np.abs(us[ux < 0])

      us[np.isnan(us) | np.isinf(us)] = 0
      un[np.isnan(un) | np.isinf(un)] = 0

      ux_avg=np.mean(ux) 
      uy_avg=np.mean(uy)
      Ub=Reb*nu/H

      print(f"#################################################")
      print(f'Ux(min) at the inclined bed: %f ' % ux.min() , 'm/s')
      print(f'Ux(max) at the inclined bed: %f ' % ux.max(), 'm/s')
      print(f"Ux average at the inclined bed: {ux_avg:.4f} m/s")
      print(f'Uy(min) at the inclined bed: %f ' % uy.min() , 'm/s')
      print(f'Uy(max) at the inclined bed: %f ' % uy.max(), 'm/s')
      print(f"Uy average at the inclined bed: {uy_avg:.4f} m/s")
      print(f"#################################################")

      #Slopewise inclined plane velocity contourmap !
      fig = plt.figure(figsize=(6, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      abs_max = np.max(np.abs(us))
      contour = plt.contourf(xclinterp/W, yclinterp/W, us, cmap=vcolormap, levels=np.linspace(-abs_max, abs_max, 100))
      quiver_spacing = 30
      q=plt.quiver(xclinterp[0,::quiver_spacing]/W, yclinterp[:,0][::quiver_spacing]/W, us[::quiver_spacing,::quiver_spacing],  uy[::quiver_spacing,::quiver_spacing],width=0.004,scale=1.5,headwidth=2,color='#808000')
      cbar = plt.colorbar(contour)
      if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
      if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
      if inst==0:
         cbar.set_label('$\\overline{U_s} (m/s)$', fontsize=12)
      else:
         cbar.set_label('${U_s} (m/s)$', fontsize=12)
      cbar.ax.tick_params(labelsize=10)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.xlabel('$x/W$', fontsize=12)
      plt.ylabel('$y/W$', fontsize=12)
      filename='-inclineplane-slopewise-velocity-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      #Spanwise inclined plane velocity contourmap !
      fig = plt.figure(figsize=(6, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'
      abs_max = np.max(np.abs(uy/Ub))
      contour = plt.contourf(xclinterp/W, yclinterp/W, uy/Ub, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
      if flagssurf == 1 and sedfoam == 0:
         #plt.plot(xcli_range / W, Lspread / W, color='black', linewidth=2, label='Lspread')
         cont = plt.contour(xclinterp / W, yclinterp / W, uy / Ub, levels=[0], colors='black', linewidths=0)
         xspread = []
         yspread = []
         for path in cont.get_paths():  
            vertices = path.vertices
            xspread.append(vertices[:, 0]) 
            yspread.append(vertices[:, 1])  
         xspread = np.concatenate(xspread)
         yspread = np.concatenate(yspread)
         filtered_xspread = []
         filtered_yspread = []
         for x_val, y_val in zip(xspread, yspread):
             if (0.5 * xp / W <= x_val <= Lxb / 2 / W) and (W / (2*W) <= y_val <= yinterpmax_clin / W):
                 filtered_xspread.append(x_val)
                 filtered_yspread.append(y_val)
         filtered_xspread = np.array(filtered_xspread)
         filtered_yspread = np.array(filtered_yspread)
         #plt.plot(filtered_xspread, filtered_yspread, color='blue', linestyle=' ', marker='o', markersize=0.5, label='filtered extracted')
         xspread=filtered_xspread*W
         Lspread=filtered_yspread*W
      quiver_spacing = 50
      q=plt.quiver(xclinterp[0,::quiver_spacing]/W, yclinterp[:,0][::quiver_spacing]/W, us[::quiver_spacing,::quiver_spacing]/Ub,  uy[::quiver_spacing,::quiver_spacing]/Ub,width=0.004,scale=20,headwidth=2,color='black')  
      cbar = plt.colorbar(contour)
      if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        ax=plt.gca()
        ax_top=ax.secondary_xaxis('top')
        ax_top.set_xticks([xp/W])
        ax_top.set_xticklabels(['$x_p$'],fontsize=24,color='black')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
      if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
      if inst==0:
         cbar.set_label('$\\overline{v}/U_0$', fontsize=28)
      else:
         cbar.set_label('${v}/U_0$', fontsize=28)
      cbar.ax.tick_params(labelsize=24)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      plt.tick_params(axis='x', labelsize=24) 
      plt.tick_params(axis='y', labelsize=24)
      cbar.ax.yaxis.set_major_formatter(formatter)
      cbar.set_ticks(np.linspace(-abs_max, +abs_max, 6))
      cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
      plt.xlabel('$x/W$', fontsize=28)
      plt.ylabel('$y/W$', fontsize=28)
      filename='-inclineplane-spanwise-velocity-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")   

      if flagssurf == 1 and sedfoam == 0:
         #Inclined plane based plume spread profile !
         fig = plt.figure(figsize=(6, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         csvname ='-x-on-inclined-plane.csv'
         csvname = sim + csvname
         pd.DataFrame(xspread, columns=['x']).to_csv(csvname, index=False)
         csvname ='-Lspread-on-inclined-plane.csv'
         csvname = sim + csvname
         pd.DataFrame(Lspread, columns=['Lspread']).to_csv(csvname, index=False)
         plt.plot(xspread/W, Lspread, linestyle=' ', marker='o', markersize=0.5, color='black')
         plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
         plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
         plt.xlabel('$x/W$', fontsize=12)
         if inst==0:
           plt.ylabel('$\\overline{L_{s}} (m)$', fontsize=12)
         else:
           plt.ylabel('$L_{s} (m)$', fontsize=12)
         filename='-curve-Lspread-on-inclined-plane.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")   

      if flagssurf == 1 and sedfoam == 0:
         #Proxy of plume ambient interface per unit streamwise distance !
         fig = plt.figure(figsize=(6, 4), dpi=500)
         plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
         plt.rcParams['xtick.direction'] = 'in'
         plt.rcParams['ytick.direction'] = 'in'
         csvname ='-x-on-inclined-plane.csv'
         csvname = sim + csvname
         pd.DataFrame(xspread, columns=['x']).to_csv(csvname, index=False)
         D=xspread*tan(beta/180*pi)+H;
         I=((Lspread)**2+D**2)**0.5
         Aint=2*I
         csvname ='-Aint-based-on-inclined-plane.csv'
         csvname = sim + csvname
         pd.DataFrame(Aint, columns=['Aint']).to_csv(csvname, index=False)
         plt.plot(xspread/W, Aint, linestyle=' ', marker='o', markersize=0.5, color='black')
         plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
         plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
         plt.xlabel('$x/W$', fontsize=12)
         if inst==0:
           plt.ylabel('$2\\overline{I}$', fontsize=12)
         else:
           plt.ylabel('$2{I}$', fontsize=12)
         filename='-proxy-of-amount-of-plume-ambient-interface-per-unit-streamwise-distance.pdf'
         savename = sim + filename
         plt.savefig(savename, bbox_inches="tight")
         plt.close()
         print(f"Plotted \"{savename}\" successfully !")      
  
   #Surface sub-grid k map !
   if flagksurf == 1 and inst != 0 and sedfoam == 0:
     print(f"############################################################################################")
     print(f"Surface sub-grid k contourmap !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     k_i = griddata((x[Iplane], y[Iplane]), k[Iplane], (xinterp, yinterp), method='linear')
      
     kclean = np.zeros((ngridy_surf, ngridx_surf))
     for i in range(ngridy_surf):
       for j in range(ngridx_surf):
         c = k_i[i,j]
         if np.isnan(c):
             kclean[i,j] = 0
         else:
             kclean[i,j] = c.item()
     
     n=25
     c=15   
     delta=5
     n = int((min(range(len(xi)), key=lambda i: abs(xi[i] - 2*W)) - c)/delta)
     jmax=np.zeros(n)
     kmax=np.zeros(n)
     for i in range (0,n):
        for j in range (0,len(k_i[:,int(i*delta+c)])):  
            if yinterp[j,int(i*delta+c)]<0:
                if k_i[j,int(i*delta+c)]>kmax[i]:
                    kmax[i]=k_i[j,int(i*delta+c)]
                    jmax[i]=j                      
     x_kmax=np.zeros(n+1)
     y_kmax=np.zeros(n+1)
     x_kmax[0]=0
     y_kmax[0]=-W
     for i in range (0,n-1):
        x_kmax[i+1]=xinterp[0,int(i*delta+c)]
        y_kmax[i+1]=yinterp[int(jmax[i]),0]
     degree=4
     func=np.polyfit(x_kmax,y_kmax,degree)
     pol = np.poly1d(func)
     yvals = pol(x_kmax) 

     #Surface sub-grid k contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(kclean))
     ax = plt.gca()
     ymin, ymax = ax.get_ylim()
     offset = 1.0 * (ymax - ymin)
     plt.contourf(xinterp/W, yinterp/W, kclean, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.text(xp/W, (ymax + offset)/W, '$99\\%R$', ha='center', va='bottom', fontsize=5)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.text(xp/W, (ymax + offset)/W, '$99\\%{\\alpha_0}$', ha='center', va='bottom', fontsize=5)
     if flagucenter == 1:
        plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.text(xd/W, (ymax + offset)/W, '$h_c$', ha='center', va='bottom', fontsize=5)
     cbar.set_label('${k} ({m}^{2}/{s}^{2})$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     stopk = next((i for i, y in enumerate(yvals) if y > 0), n-1)
     stopk = np.argmin(np.abs(yvals))
     plt.plot(x_kmax[:stopk]/W,yvals[:stopk]/W,color='black')
     plt.plot(x_kmax[:stopk]/W,-yvals[:stopk]/W,color='black')
     plt.axvline(x=x_kmax[stopk]/W, color='black', linestyle='--', linewidth=1)
     plt.text(x_kmax[stopk]/W, (ymax + offset)/W, '$k$', ha='center', va='bottom', fontsize=5)
     filename='-surfaceplane-k-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Surface instanteneous tke contourmap !
     Ux_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[0, Iplane]), (xinterp, yinterp), method='linear')
     Uy_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[1, Iplane]), (xinterp, yinterp), method='linear')
     Uz_i = griddata((x[Iplane], y[Iplane]), np.transpose(U[2, Iplane]), (xinterp, yinterp), method='linear')
     Umean = readvector(sol, timename, f'UMean_w{window}U')
     Umeanx_i = griddata((x[Iplane], y[Iplane]), np.transpose(Umean[0, Iplane]), (xinterp, yinterp), method='linear')
     Umeany_i = griddata((x[Iplane], y[Iplane]), np.transpose(Umean[1, Iplane]), (xinterp, yinterp), method='linear')
     Umeanz_i = griddata((x[Iplane], y[Iplane]), np.transpose(Umean[2, Iplane]), (xinterp, yinterp), method='linear')
     Uprimex_i=Ux_i-Umeanx_i
     Uprimey_i=Uy_i-Umeany_i
     Uprimez_i=Uz_i-Umeanz_i
     TKEinst=Uprimex_i**2+Uprimey_i**2+Uprimez_i**2

     TKEinstclean = np.zeros((ngridy_surf, ngridx_surf))
     for i in range(ngridy_surf):
       for j in range(ngridx_surf):
         c = TKEinst[i,j]
         if np.isnan(c):
             TKEinstclean[i,j] = 0
         else:
             TKEinstclean[i,j] = c.item()
     
     n=25
     c=15   
     delta=5
     n = int((min(range(len(xi)), key=lambda i: abs(xi[i] - 2*W)) - c)/delta)
     jmax=np.zeros(n)
     TKEinstmax=np.zeros(n)
     for i in range (0,n):
        for j in range (0,len(TKEinst[:,int(i*delta+c)])):  
            if yinterp[j,int(i*delta+c)]<0:
                if TKEinst[j,int(i*delta+c)]>TKEinstmax[i]:
                    TKEinstmax[i]=TKEinst[j,int(i*delta+c)].item()
                    jmax[i]=j                      
     x_TKEinstmax=np.zeros(n+1)
     y_TKEinstmax=np.zeros(n+1)
     x_TKEinstmax[0]=0
     y_TKEinstmax[0]=-W
     for i in range (0,n-1):
        x_TKEinstmax[i+1]=xinterp[0,int(i*delta+c)]
        y_TKEinstmax[i+1]=yinterp[int(jmax[i]),0]
     degree=4
     func=np.polyfit(x_TKEinstmax,y_TKEinstmax,degree)
     pol = np.poly1d(func)
     yvals = pol(x_TKEinstmax)

     #Surface instanteneous tke contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(TKEinstclean))
     ax = plt.gca()
     ymin, ymax = ax.get_ylim()
     offset = 1.0 * (ymax - ymin)
     plt.contourf(xinterp/W, yinterp/W, TKEinstclean, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.text(xp/W, (ymax + offset)/W, '$99\\%R$', ha='center', va='bottom', fontsize=5)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.text(xp/W, (ymax + offset)/W, '$99\\%{\\alpha_0}$', ha='center', va='bottom', fontsize=5)
     if flagucenter == 1:
        plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.text(xd/W, (ymax + offset)/W, '$h_c$', ha='center', va='bottom', fontsize=5)
     cbar.set_label('${TKE} ({m}^{2}/{s}^{2})$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     stopTKEinst = next((i for i, y in enumerate(yvals) if y > 0), n-1)
     stopTKEinst = np.argmin(np.abs(yvals))
     plt.plot(x_TKEinstmax[:stopTKEinst]/W,yvals[:stopTKEinst]/W,color='black')
     plt.plot(x_TKEinstmax[:stopTKEinst]/W,-yvals[:stopTKEinst]/W,color='black')
     plt.axvline(x=x_TKEinstmax[stopTKEinst]/W, color='black', linestyle='--', linewidth=1)
     plt.text(x_TKEinstmax[stopTKEinst]/W, (ymax + offset)/W, '$TKE$', ha='center', va='bottom', fontsize=5)
     filename='-surfaceplane-tke-inst-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")
     
     perdiff=abs(x_kmax[stopk]-x_TKEinstmax[stopTKEinst])/x_TKEinstmax[stopTKEinst]*100

     print(f"##########################################################################################################")
     print(f"Location where distance of k-max from the centerline is minimum: {x_kmax[stopk]:.5f} (m)")
     print(f"Location where distance of TKEinst-max from the centerline is minimum: {x_TKEinstmax[stopTKEinst]:.5f} (m)")
     print(f"Percentage difference of the location where distance from the centerline is minimum: {perdiff:.2f} %")
     print(f"##########################################################################################################")
     
     masktkelarge = TKEinstclean > 1e-4
     ratio = (TKEinstclean[masktkelarge] / (TKEinstclean[masktkelarge] + kclean[masktkelarge]))
     minper=np.min(ratio)*100
     maxper=np.max(ratio)*100
     avper=np.mean(ratio)*100
     total = TKEinstclean[masktkelarge].size
     count90per = np.sum(ratio*100 > 90)
     fraction90per = count90per / total * 100
     count20per = np.sum(ratio*100 > 20)
     fraction20per = count20per / total * 100

     print(f"##########################################################################################################")
     print(f"Min percentage of the total local TKE > 1e-4 m^2/s^2 on the surface plane that is resolved: {minper:.2f} %")
     print(f"Max percentage of the total local TKE > 1e-4 m^2/s^2 on the surface plane that is resolved: {maxper:.2f} %")
     print(f"Average percentage of the total local TKE > 1e-4 m^2/s^2 on the surface plane that is resolved: {avper:.2f} %")
     print(f"Percentage of points at the surfaceplane where resolved TKE > 1e-4 m^2/s^2 is greater than 90 %: {fraction90per:.2f} %")
     print(f"Percentage of points at the surfaceplane where resolved TKE > 1e-4 m^2/s^2 is greater than 20 %: {fraction20per:.2f} %")
     print(f"##########################################################################################################")

     #Surface total tke contourmap !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     abs_max = np.nanmax(np.abs(kclean+TKEinstclean))
     ax = plt.gca()
     ymin, ymax = ax.get_ylim()
     offset = 1.0 * (ymax - ymin)
     plt.contourf(xinterp/W, yinterp/W, kclean+TKEinstclean, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
     cbar = plt.colorbar()
     if flagssurf == 1 and sedfoam==0:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.text(xp/W, (ymax + offset)/W, '$99\\%R$', ha='center', va='bottom', fontsize=5)
     if flagasurf == 1 and sedfoam==1:
        plt.axvline(x=xp/W, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.text(xp/W, (ymax + offset)/W, '$99\\%{\\alpha_0}$', ha='center', va='bottom', fontsize=5)
     if flagucenter == 1:
        plt.axvline(x=xd/W, color='black', linestyle=':', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.text(xd/W, (ymax + offset)/W, '$h_c$', ha='center', va='bottom', fontsize=5)
     cbar.set_label('${TKE+k} ({m}^{2}/{s}^{2})$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('$x/W$', fontsize=12)
     plt.ylabel('$y/W$', fontsize=12)
     stopTKEinst = next((i for i, y in enumerate(yvals) if y > 0), n-1)
     stopTKEinst = np.argmin(np.abs(yvals))
     plt.plot(x_TKEinstmax[:stopTKEinst]/W,yvals[:stopTKEinst]/W,color='black')
     plt.plot(x_TKEinstmax[:stopTKEinst]/W,-yvals[:stopTKEinst]/W,color='black')
     plt.axvline(x=x_TKEinstmax[stopTKEinst]/W, color='black', linestyle='--', linewidth=1)
     plt.text(x_TKEinstmax[stopTKEinst]/W, (ymax + offset)/W, '$TKE$', ha='center', va='bottom', fontsize=5)
     filename='-surfaceplane-total-tke-field.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

   #Centerplane sub-grid k map !
   if flagksurf == 1 and inst != 0 and sedfoam == 0:
      print(f"############################################################################################")
      print(f"Centerplane k contourmap !")
      print(f"############################################################################################")
      #Centerplane k contourmap !
      xi = np.linspace(xinterpmin_center, xinterpmax_center, ngridx_center)
      zi = np.linspace(zinterpmin_center, zinterpmax_center, ngridz_center)
      xinterp, zinterp = np.meshgrid(xi, zi)
      Iplane=np.where(np.logical_and(y>=Yplane_center-dy_center,y<=Yplane_center+dy_center))
      k_i = griddata((x[Iplane], z[Iplane]), k[Iplane], (xinterp, zinterp), method='linear')
      
      kclean = np.zeros((ngridz_center,ngridx_center))
      for i in range (0,ngridz_center):
         for j in range (0,ngridx_center):
            c = k_i[i,j]
            if np.isnan(c):
                kclean[i,j] = 0
            else:
                kclean[i,j] = c.item()

      #Centerplane instanteneous tke contourmap !
      Ux_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[0, Iplane]), (xinterp, zinterp), method='linear')
      Uy_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[1, Iplane]), (xinterp, zinterp), method='linear')
      Uz_i = griddata((x[Iplane], z[Iplane]), np.transpose(U[2, Iplane]), (xinterp, zinterp), method='linear')
      Umean = readvector(sol, timename, f'UMean_w{window}U')
      Umeanx_i = griddata((x[Iplane], z[Iplane]), np.transpose(Umean[0, Iplane]), (xinterp, zinterp), method='linear')
      Umeany_i = griddata((x[Iplane], z[Iplane]), np.transpose(Umean[1, Iplane]), (xinterp, zinterp), method='linear')
      Umeanz_i = griddata((x[Iplane], z[Iplane]), np.transpose(Umean[2, Iplane]), (xinterp, zinterp), method='linear')
      Uprimex_i=Ux_i-Umeanx_i
      Uprimey_i=Uy_i-Umeany_i
      Uprimez_i=Uz_i-Umeanz_i
      TKEinst=Uprimex_i**2+Uprimey_i**2+Uprimez_i**2

      TKEinstclean = np.zeros((ngridz_center,ngridx_center))
      for i in range (0,ngridz_center):
            for j in range (0,ngridx_center):
               c = TKEinst[i,j]
               if np.isnan(c):
                  TKEinstclean[i,j] = 0
               else:
                  TKEinstclean[i,j] = c.item()

      #Centerplane sub-grid k contourmap !
      fig = plt.figure(figsize=(16, 4), dpi=500)
      plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
      plt.rcParams['xtick.direction'] = 'in'
      plt.rcParams['ytick.direction'] = 'in'          
      xs=np.array([0, xinterpmax_center])
      ys=np.array([-H,-H-xinterpmax_center*tan(beta/180*pi)])
      plt.plot(xs/W,ys/H,c='black') 
      abs_max = np.nanmax(np.abs(kclean))
      plt.contourf(xinterp/W, zinterp/H, kclean, cmap=vcolormap, levels=np.linspace(-abs_max, +abs_max, 100))
      cbar = plt.colorbar()
      cbar.set_label('${k} ({m}^{2}/{s}^{2})$', fontsize=12)
      cbar.ax.tick_params(labelsize=10)
      formatter = ScalarFormatter(useMathText=True)
      formatter.set_powerlimits((0, 0)) 
      cbar.ax.yaxis.set_major_formatter(formatter)
      plt.xlabel('$x/W$', fontsize=28)
      plt.ylabel('$z/H_0$', fontsize=28)
      plt.ylim((-H-xinterpmax_center*tan(beta/180*pi))/H,0)
      filename='-centerplane-k-field.pdf'
      savename = sim + filename
      plt.savefig(savename, bbox_inches="tight")
      plt.close()
      print(f"Plotted \"{savename}\" successfully !")

      masktkelarge = TKEinstclean > 1e-4
      ratio = (TKEinstclean[masktkelarge] / (TKEinstclean[masktkelarge] + kclean[masktkelarge]))
      minper=np.min(ratio)*100
      maxper=np.max(ratio)*100
      avper=np.mean(ratio)*100
      total = TKEinstclean[masktkelarge].size
      count90per = np.sum(ratio*100 > 90)
      fraction90per = count90per / total * 100
      count20per = np.sum(ratio*100 > 20)
      fraction20per = count20per / total * 100

      print(f"##########################################################################################################")
      print(f"Min percentage of the total local TKE > 1e-4 m^2/s^2 on the centerplane that is resolved: {minper:.2f} %")
      print(f"Max percentage of the total local TKE > 1e-4 m^2/s^2 on the centerplane that is resolved: {maxper:.2f} %")
      print(f"Average percentage of the total local TKE > 1e-4 m^2/s^2 on the centerplane that is resolved: {avper:.2f} %")
      print(f"Percentage of points at the centerplane where resolved TKE > 1e-4 m^2/s^2 is greater than 90 %: {fraction90per:.2f} %")
      print(f"Percentage of points at the centerplane where resolved TKE > 1e-4 m^2/s^2 is greater than 20 %: {fraction20per:.2f} %")
      print(f"##########################################################################################################")

   #Kelvin-Helmholtz waves !
   if flagkhwave == 1 and inst != 0 and sedfoam == 0:
     print(f"############################################################################################")
     print(f"Surface Kelvin-Helmholtz waves !")
     print(f"############################################################################################")
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     T_i = griddata((x[Iplane], y[Iplane]), T[Iplane], (xinterp, yinterp), method='linear')

     for m in range (0,len(T_i[ngridy_surf//2])):
       if T_i[ngridy_surf//2][m]<=0.99*R:
          xpinst=xinterp[0][m]
          break

     aptriangle = W/2;
     prostriangle = xpinst;
     hyptriangle = ((W/2)**2+prostriangle**2)**0.5
     cosangle = prostriangle / hyptriangle
     angle = math.acos(cosangle)

     maskone = yinterp > 0 
     masktwo = np.isclose(T_i/R, 0.99, atol=1e-3)
     mask = maskone & masktwo
     trindexes = np.where(mask)
     trix = xinterp[trindexes]
     triy = yinterp[trindexes]
     data = pd.DataFrame({'trix': trix, 'triy': triy})
     data = data.drop_duplicates(subset=['trix', 'triy'])
     data.sort_values('trix', inplace=True)
     trix = data['trix'].to_numpy()
     triy = data['triy'].to_numpy()
     trix = trix[trix <= xpinst]
     triy = triy[:len(trix)]
     yedge = (prostriangle-trix) * math.tan(angle)
     uptris = (trix - prostriangle) / math.cos(angle) + hyptriangle
     uptrin = (triy - yedge) * math.sin(math.pi / 2 - angle)

     uniqueuptris, uniqueindices = np.unique(uptris, return_index=True)
     uniqueuptrin = uptrin[uniqueindices]
     npoints = len(uniqueuptris)  
     uniformuptris = np.linspace(uniqueuptris.min(), uniqueuptris.max(), npoints)
     interpolator = interp1d(uniqueuptris, uniqueuptrin, kind='cubic') 
     uniformuptrin = interpolator(uniformuptris)
     
     maskone = yinterp < 0 
     masktwo = np.isclose(T_i/R, 0.99, atol=1e-3)
     mask = maskone & masktwo
     trindexes = np.where(mask)
     trix = xinterp[trindexes]
     triy = - yinterp[trindexes]
     data = pd.DataFrame({'trix': trix, 'triy': triy})
     data = data.drop_duplicates(subset=['trix', 'triy'])
     data.sort_values('trix', inplace=True)
     trix = data['trix'].to_numpy()
     triy = data['triy'].to_numpy()
     trix = trix[trix <= xpinst]
     triy = triy[:len(trix)]
     yedge = (prostriangle-trix) * math.tan(angle)
     downtris = (trix - prostriangle) / math.cos(angle) + hyptriangle
     downtrin = (triy - yedge) * math.sin(math.pi / 2 - angle)

     #Kelvin-Helmholtz waves curve !
     fig = plt.figure(figsize=(16, 4), dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-kh-wave-x-coordinate.csv'
     csvname = sim + csvname
     pd.DataFrame(uniformuptris, columns=['s']).to_csv(csvname, index=False)
     csvname ='-kh-wave-y-coordinate.csv'
     csvname = sim + csvname
     pd.DataFrame(uniformuptrin, columns=['KHwave']).to_csv(csvname, index=False)  
     ttindex = int(len(uniformuptrin) * 2 / 3)        
     plt.plot(uptris[:ttindex], uptrin[:ttindex], '-', linewidth=2, color='black', label=f'{sim}-left-side')
     plt.plot(uniformuptris, uniformuptrin, '-', linewidth=2, color='black', linestyle='--', label=f'{sim}-left-side-interpolated')
     # plt.plot(uniformuptris[0], uniformuptrin[0], 'o', color='red', markersize=8, label='First Point')
     # plt.plot(uniformuptris[-1], uniformuptrin[-1], 'o', color='red', markersize=8, label='Last Point')
     plt.plot(downtris[:ttindex], downtrin[:ttindex], '-', linewidth=2, color='red', label=f'{sim}-right-side')
     plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     plt.xlabel('$s (m)$', fontsize=12)
     plt.ylabel('$\\eta (m)$', fontsize=12)
     plt.xlim(0, hyptriangle)
     plt.ylim(-W/4, W/2)
     plt.legend()
     ax_inset = inset_axes(plt.gca(), width="15%", height="40%", loc='upper left', borderpad=2) 
     contour = ax_inset.contourf(xinterp, yinterp, T_i/R, cmap=scolormap, levels=np.linspace(0, np.max(T_i/R), 100))
     ax_inset.contour(xinterp, yinterp, T_i/R, levels=[0.99], colors='black', linewidths=2)
     ax_inset.plot(xpinst, 0, 'ro', markersize=4) 
     ax_inset.plot([xpinst, 0], [0, -W/2], 'r--', linewidth=1)  
     ax_inset.plot([xpinst, 0], [0, W/2], 'r--', linewidth=1)  
     ax_inset.axhline(y=0, color='black', linestyle='--', linewidth=1)
     filename='-curve-kelvin-helmholtz-wave.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")
      
     signal = uniformuptrin[:ttindex]    
     spectrum = np.fft.fft(signal)
     n = len(signal) 
     ds = uniformuptris[1] - uniformuptris[0]
     sfreq = np.fft.fftfreq(n, d= ds)
     ks = 2 * np.pi * sfreq
     asd = np.abs(spectrum)
     esd = (asd**2) / n  
     ks = ks[1:]  
     esd = esd[1:] 

     esdmax = np.max(esd)
     ksmax_index = np.argmax(esd)
     ksmax = ks[ksmax_index]
     lambdasmax =  2 * np.pi / ksmax

     #Kelvin-Helmholtz spectrum curve !
     fig = plt.figure(figsize=(6, 4), dpi=500)
     ax = plt.gca()
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-kh-spectrum-wavenumber.csv'
     csvname = sim + csvname
     pd.DataFrame(ks[:(n-1) // 2] / ksmax, columns=['ks']).to_csv(csvname, index=False)
     csvname ='-kh-spectrum-energy-density.csv'
     csvname = sim + csvname
     pd.DataFrame(esd[:(n-1) // 2], columns=['KHspectrum']).to_csv(csvname, index=False)
     plt.plot(ks[:(n-1) // 2] / ksmax, esd[:(n-1) // 2], linewidth=2, color='black', label=f'{sim}') 
     plt.xlabel('$k_s / k_p$')
     plt.ylabel('$E(k_s) (m^3)$')
     lambdasmaxtext = f"$\\lambda_c$ = {lambdasmax:.3f} m"
     plt.text(0.95, 0.95, lambdasmaxtext, ha='right', va='top', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='lightgrey', alpha=0.5))
     filename='-curve-kelvin-helmholtz-spectrum.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     print(f"###################################################################################################################")
     print(f"Maximum energy spectral density: {esdmax:.5f} (m^3)")
     print(f"Wavenumber ks where maximum energy spectral density is concentrated : {ksmax:.5f} (1/m)")
     print(f"Corresponding wavelength lambdac where maximum energy spectral density is concentrated : {lambdasmax:.5f} (m)")
     print(f"###################################################################################################################")

   #River-Lake 3D interface reconstruction !
   if flagriverlake3Dinterface == 1:
     print(f"############################################################################################")
     print(f"River-Lake 3D interface reconstruction !")
     print(f"############################################################################################")

     x_flat = x.flatten()
     y_flat = y.flatten()
     z_flat = z.flatten()
     if sedfoam==0:
        T_flat = T.flatten()
        tolerance=1e-4
        isolevel=0.99
        mask = (np.abs(T_flat - isolevel*R) < tolerance) & (x_flat > 0)   

        interfacepoints = np.vstack((x_flat[mask], y_flat[mask], z_flat[mask])).T
        hull = ConvexHull(interfacepoints)
        area = hull.area
     else:
        alpha_flat = alpha.flatten()
        tolerance=1e-4
        isolevel=0.75
        mask = (np.abs(alpha_flat - isolevel*a0) < tolerance) & (x_flat > 0)   

        interfacepoints = np.vstack((x_flat[mask], y_flat[mask], z_flat[mask])).T
        hull = ConvexHull(interfacepoints)
        area = hull.area

     print(f"############################################################################################")
     print(f"Estimated river-lake interface surface area: {area:.2f} m^2")
     print(f"############################################################################################")

     #Overview (z+ to z-)
     fig = plt.figure(figsize=(6, 4), dpi=500)
     ax = plt.axes(projection="3d")
     ax.scatter(x_flat[mask], y_flat[mask], z_flat[mask], color='#add8e6', alpha=0.5, s=1)
     for simplex in hull.simplices:
         ax.plot_trisurf(interfacepoints[simplex, 0], interfacepoints[simplex, 1], interfacepoints[simplex, 2], color='gray', alpha=0.3, linewidth=0.5, antialiased=True)
     ax.set_xlabel("x (m)")
     ax.set_ylabel("y (m)")
     ax.set_zlabel("z (m)")
     plt.title(f"3D Interface - {isolevel*100} % Iso-Contour -  Overview")
     filename = '-river-lake-3D-interface-overview.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully!")

     #Top View (z+ to z-)
     fig = plt.figure(figsize=(6, 4), dpi=500)
     ax = plt.axes(projection="3d")
     ax.scatter(x_flat[mask], y_flat[mask], z_flat[mask], color='#add8e6', alpha=0.5, s=1)
     ax.set_xlabel("x (m)")
     ax.set_ylabel("y (m)")
     ax.set_zticks([]) 
     ax.set_zticklabels([]) 
     plt.title(f"3D Interface - {isolevel*100} % Iso-Contour - Top View")
     ax.view_init(elev=90, azim=0)  
     filename = '-river-lake-3D-interface-top-view.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully!")

     #Side View (y+ to y-)
     fig = plt.figure(figsize=(6, 4), dpi=500)
     ax = plt.axes(projection="3d")
     ax.scatter(x_flat[mask], y_flat[mask], z_flat[mask], color='#add8e6', alpha=0.5, s=1)
     ax.set_xlabel("x (m)")
     ax.set_zlabel("z (m)")
     ax.set_yticks([])  
     ax.set_yticklabels([])  
     plt.title(f"3D Interface - {isolevel*100} % Iso-Contour - Side View")
     ax.view_init(elev=0, azim=-90)  # Side view (y+ to y-)
     filename = '-river-lake-3D-interface-side-view.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully!")

     #Front View (x+ to x-)
     fig = plt.figure(figsize=(6, 4), dpi=500)
     ax = plt.axes(projection="3d")
     ax.scatter(x_flat[mask], y_flat[mask], z_flat[mask], color='#add8e6', alpha=0.5, s=1)
     ax.set_ylabel("y (m)")
     ax.set_zlabel("z (m)")
     ax.set_xticks([]) 
     ax.set_xticklabels([]) 
     plt.title(f"3D Interface - {isolevel*100} % Iso-Contour - Front View")
     ax.view_init(elev=0, azim=0) 
     filename = '-river-lake-3D-interface-front-view.pdf'
     savename = sim + filename
     plt.savefig(savename, bbox_inches="tight")
     plt.close()
     print(f"Plotted \"{savename}\" successfully!")

   #Theoretical model prediction of plunging point !
   if flagmodel == 1 and sedfoam==0:
     print(f"############################################################################################")
     print(f"Theoretical model prediction of the plunge curve !")
     print(f"############################################################################################")

     #Simulation 99% salinity based plunging point
     xi = np.linspace(xinterpmin_surf, xinterpmax_surf, ngridx_surf)
     yi = np.linspace(yinterpmin_surf, yinterpmax_surf, ngridy_surf)
     xinterp, yinterp = np.meshgrid(xi, yi)
     Iplane=np.where(np.logical_and(z>=Zvert_surf-dz_surf,z<=Zvert_surf+dz_surf))
     T_i = griddata((x[Iplane], y[Iplane]), T[Iplane], (xinterp, yinterp), method='linear')

     for m in range (0,len(T_i[ngridy_surf//2])):
       if T_i[ngridy_surf//2][m]<=0.99*R:
          xp=xinterp[0][m]
          break

     #Linear model
     Cf =0.78;
     initialguess=xp;
     uthl=(Reb*nu/H)*Cf
     solution = fsolve(lambda xgthl: xgthl - W/2*uthl/((R*g*(H + xgthl*np.tan(beta/180*pi)))**0.5 / 2), initialguess)
     Cs = 5.5;
     Cs = 6.14;
     Cs = 4.1;
     Cs = 5.23;
     solution = fsolve(lambda xgthl: xgthl - W/2*uthl/((R*g*(H + Cs*xgthl*np.tan(beta/180*pi)))**0.5 / 2), initialguess) 
     # Cs = 0.47;
     # solution = fsolve(lambda xgthl: xgthl - Cs*W/2*uthl/((R*g*(H + xgthl*np.tan(beta/180*pi)))**0.5 / 2), initialguess) 
     xgthl = solution[0]
     deq = (H + Cs*xgthl*np.tan(beta/180*pi)) / (H + xgthl*np.tan(beta/180*pi))
     
     #Nonlinear model 
     Cf =0.8;
     Q = Reb * nu * W
     uthnl = Cf * Reb * nu / H  
     dtthnl = 2 * xgthl / uthnl / 100 
     n_steps = int(2 * xgthl / uthnl / dtthnl)  
     tthnl = np.linspace(0, 2 * xgthl / uthnl, n_steps) 
     ythnl = np.zeros(n_steps)  
     xthnl = np.zeros(n_steps)  
     vthnl = np.zeros(n_steps)  
     vcont = np.zeros(n_steps)  
     ythnl[0] = - W / 2 
     xthnl[0] = 0
     Wxold = W 
     for i in range(1, n_steps):
       xthnl[i] = xthnl[i - 1] + uthnl * dtthnl
       Hx = H + xthnl[i - 1] * np.tan(beta / 180 * np.pi)
       Delta = (Q**2/(H*W)-g*H**2*W/2.)**2+2*g*Hx*Q**2
       Wx = (-(Q**2/(H*W)-g*H**2*W/2.) + np.sqrt(Delta))/(g*Hx**2)
       vcont[i - 1] = -(Wx/2 - Wxold/2)/(xthnl[i]-xthnl[i - 1])*uthnl
       vthnl[i - 1] = np.sqrt(R * g * Hx) / 2 + vcont[i - 1]
       #nlfactor = 2.5
       #vthnl[i - 1] *= nlfactor 
       ythnl[i] = ythnl[i - 1] + vthnl[i - 1] * dtthnl  
       Wxold = Wx

     mask = ythnl <= 0
     xthnlfinal = xthnl[mask]
     ythnlfinal = ythnl[mask]
     closest = np.argmin(np.abs(ythnl))
     xgthnl = xthnl[closest]

     #Linear theoretical versus numerical plunge curve !
     fig = plt.figure(figsize=(6, 4),dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     plt.contourf(xinterp, yinterp, T_i/R, cmap=scolormap, levels=np.linspace(0, np.max(T_i/R), 100))
     cbar = plt.colorbar()
     plt.axvline(x=xp, color='black', linestyle='--', linewidth=1)
     plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     cont = plt.contour(xinterp, yinterp, T_i/R, levels=[0.99], colors='red', linewidths=2)
     xcont = []
     ycont = []
     for path in cont.get_paths():  
        vertices = path.vertices
        xcont.append(vertices[:, 0]) 
        ycont.append(vertices[:, 1])  
     xcont = np.concatenate(xcont)
     ycont = np.concatenate(ycont)
     xcont, ycont = np.unique(np.column_stack((xcont, ycont)), axis=0).T
     plt.plot(xcont, ycont, color='red', linestyle=' ', marker='o', markersize=0.5, label='simulation')
     csvname = '-x-plunge-curve-simulation.csv'
     csvname = sim + csvname
     pd.DataFrame(xcont, columns=['x-plunge-curve']).to_csv(csvname, index=False)
     csvname = '-y-plunge-curve-simulation.csv'
     csvname = sim + csvname
     pd.DataFrame(ycont, columns=['y-plunge-curve']).to_csv(csvname, index=False)
     plt.plot([0, xgthl], [-W/2, 0], color='green', linestyle='--', linewidth=2, label='linear model')
     plt.plot([0, xgthl], [W/2, 0], color='green', linestyle='--', linewidth=2, label='_nolegend_')
     csvname = '-x-linear-model.csv'
     csvname = sim + csvname
     pd.DataFrame([0, xgthl], columns=['x-lmodel']).to_csv(csvname, index=False)
     csvname = '-y-linear-model.csv'
     csvname = sim + csvname
     pd.DataFrame([-W/2, 0], columns=['y-lmodel']).to_csv(csvname, index=False)
     if inst==0:
        cbar.set_label('$\\overline{R}$/${R}_{0}$', fontsize=12)
     else:
        cbar.set_label('${R}$/${R}_{0}$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('x (m)', fontsize=12)
     plt.ylabel('y (m)', fontsize=12)
     plt.legend(loc='upper right')
     filename='-linear-theoretical-versus-numerical.pdf'
     savename = sim + filename
     plt.savefig(savename)
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     #Nonlinear theoretical versus numerical plunge curve !
     fig = plt.figure(figsize=(6, 4),dpi=500)
     plt.rcParams.update({'font.size': 11, 'font.family': 'serif','font.serif': ['Computer Modern Roman'],  'text.usetex': True})
     plt.rcParams['xtick.direction'] = 'in'
     plt.rcParams['ytick.direction'] = 'in'
     csvname ='-x-nonlinear-model.csv'
     csvname = sim + csvname
     pd.DataFrame(xthnlfinal, columns=['x-nlmodel']).to_csv(csvname, index=False)
     csvname ='-y-nonlinear-model.csv'
     csvname = sim + csvname
     pd.DataFrame(ythnlfinal, columns=['y-nlmodel']).to_csv(csvname, index=False)
     plt.contourf(xinterp, yinterp, T_i/R, cmap=scolormap, levels=np.linspace(0, np.max(T_i/R), 100))
     cbar = plt.colorbar()
     plt.axvline(x=xp, color='black', linestyle='--', linewidth=1)
     plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
     cont = plt.contour(xinterp, yinterp, T_i/R, levels=[0.99], colors='red', linewidths=2)
     xcont = []
     ycont = []
     for path in cont.get_paths():  
        vertices = path.vertices
        xcont.append(vertices[:, 0]) 
        ycont.append(vertices[:, 1])  
     xcont = np.concatenate(xcont)
     ycont = np.concatenate(ycont)
     xcont, ycont = np.unique(np.column_stack((xcont, ycont)), axis=0).T
     plt.plot(xcont, ycont, color='green', linestyle=' ', marker='o', markersize=0.5, label='simulation')
     csvname = '-x-plunge-curve-simulation.csv'
     csvname = sim + csvname
     pd.DataFrame(xcont, columns=['x-plunge-curve']).to_csv(csvname, index=False)
     csvname = '-y-plunge-curve-simulation.csv'
     csvname = sim + csvname
     pd.DataFrame(ycont, columns=['y-plunge-curve']).to_csv(csvname, index=False)
     plt.plot(xthnlfinal, ythnlfinal , color='blue', linestyle='--', linewidth=1, label='nonlinear model')
     plt.plot(xthnlfinal, -ythnlfinal , color='blue', linestyle='--', linewidth=1, label='_nolegend_')
     if inst==0:
        cbar.set_label('$\\overline{R}$/${R}_{0}$', fontsize=12)
     else:
        cbar.set_label('${R}$/${R}_{0}$', fontsize=12)
     cbar.ax.tick_params(labelsize=10)
     formatter = ScalarFormatter(useMathText=True)
     formatter.set_powerlimits((0, 0)) 
     cbar.ax.yaxis.set_major_formatter(formatter)
     plt.xlabel('x (m)', fontsize=12)
     plt.ylabel('y (m)', fontsize=12)
     plt.legend(loc='upper right')
     filename='-nonlinear-theoretical-versus-numerical.pdf'
     savename = sim + filename
     plt.savefig(savename)
     plt.close()
     print(f"Plotted \"{savename}\" successfully !")

     print(f"######################################################################################################################")
     print(f'Plunging point position based on 99%R criterion for time={timename} x_p={xp:.5f} m')
     print(f'Plunging point position based on 99%R criterion for time={timename} normalized by the channel width x_p/W={(xp/W):.5f}')
     print(f'Plunging point position based on linear theoretical model for time={timename} x_pth={xgthl:.5f} m')
     print(f'Plunging point position based on linear theoretical model for time={timename} normalized by the channel width x_pth/W={(xgthl/W):.5f}')
     print(f'Equivalent lock-release depth for linear model for time={timename} = {(deq):.5f}Hp')
     print(f'Plunging point position based on nonlinear theoretical model for time={timename} x_pthnl={xgthnl:.5f} m')
     print(f'Plunging point position based on nonlinear theoretical model for time={timename} normalized by the channel width x_pthnl/W={(xgthnl/W):.5f}')
     print(f"######################################################################################################################")


#Timer off
endtime = time.time()
elapsedtime = endtime - starttime
print(f"############################################################################################")
print(f"Elapsed time: {elapsedtime:.2f} seconds")
print(f"############################################################################################")