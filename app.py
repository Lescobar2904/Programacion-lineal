# By Alberto Caro S.
# Ing. Civil en Informatica
# Doctor(c) Cs. de la Computacion
# Pontificia Universidad Catolica de Chile
# Programacion de Robot -> INFO1139
#---------------------------------------------------------------------
#__________      ___.           __  .__
#\______   \ ____\_ |__   _____/  |_|__| ____ _____
# |       _//  _ \| __ \ /  _ \   __\  |/ ___\\__  \
# |    |   (  <_> ) \_\ (  <_> )  | |  \  \___ / __ \_
# |____|_  /\____/|___  /\____/|__| |__|\___  >____  /
#        \/           \/                    \/     \/
#---------------------------------------------------------------------
import pygame as PG, time as Ti, random as RA, ctypes as CT
from pygame.locals import *

#---------------------------------------------------------------------
# Definicion de Constantes y Variables
#---------------------------------------------------------------------
nRES = (1184,576); nT_WX = nT_HY = 32 ; nMAX_ROBOTS = 01 ; lGo = True
nMx  = nMy = 0; nR_1 = 1154 ; nR_2 = 32

#---------------------------------------------------------------------
# Definicion Structura Robots
#---------------------------------------------------------------------
class eRobot(CT.Structure):
 _fields_ = [
             ('nF',CT.c_short),
             ('nX',CT.c_short),
             ('nY',CT.c_short),
             ('nR',CT.c_short),
             ('nS',CT.c_short),
             ('dX',CT.c_short),
             ('dY',CT.c_short),
	   ('nV',CT.c_short),
	   ('nC',CT.c_short)
            ]

#---------------------------------------------------------------------
# Definicion Structure Celda Inteligente del Mapa
#---------------------------------------------------------------------
class eCelda(CT.Structure):
 _fields_ = [
             ('nT',CT.c_ubyte), # Tipo de Tile/Baldosa
             ('nD',CT.c_ubyte), # Tile Disponible?
             ('nS',CT.c_ubyte), # 0 : No se pinta - # 1 : Si se pinta
             ('nF',CT.c_ubyte), # Fila de Mapa
             ('nC',CT.c_ubyte), # Columna de Mapa
             ('nR',CT.c_ubyte), # Recurso a Explotar:
                                # 1:Acero
                                # 2:Cobre
                                # 3:Litio
                                # 4:Butano
             ('nQ',CT.c_ubyte)  # Cantidad del Recurso
            ]

#---------------------------------------------------------------------
# Carga imagenes y convierte formato PyGame
#---------------------------------------------------------------------
def Load_Image(sFile,transp = False):
    try: image = PG.image.load(sFile)
    except PG.error,message:
           raise SystemExit,message
    image = image.convert()
    if transp:
       color = image.get_at((0,0))
       image.set_colorkey(color,RLEACCEL)
    return image

#---------------------------------------------------------------------
# Inicializa PGs.-
#---------------------------------------------------------------------
def Init_PyGame():
    PG.init()
    PG.mouse.set_visible(False)
    PG.display.set_caption(' Mapa Inteligente 2D Robotica - By Alberto Caro')
    return PG.display.set_mode(nRES)

#---------------------------------------------------------------------
# Inicilaiza parametros de los Robots
#---------------------------------------------------------------------
def Init_Robot():
    for i in range(0,nMAX_ROBOTS):
     aBoe[i].nF = 1    # Robot Tipo 1
     aBoe[i].nX = 0    # (RA.randint(0,nRES[0] - nT_WX) / nT_WX) * nT_WX
     aBoe[i].nY = 0  # (RA.randint(0,nRES[1] - nT_HY) / nT_HY) * nT_HY
     aBoe[i].nR = nR_1 # (RA.randint(0,nRES[0] - nT_WX) / nT_WX) * nT_WX
     aBoe[i].nS = 1    # Switch por defecto
     aBoe[i].dX = 1    # Por defecto robot Direccion Este.-
     aBoe[i].dY = 0
     aBoe[i].nV = 1
     aBoe[i].nC = 1
    return

#---------------------------------------------------------------------
# Inicilaiza parametros de los Robots
#---------------------------------------------------------------------
def Init_Robot_Random():
    for i in range(0,nMAX_ROBOTS):
     aBoe[i].nF = 1 # Robot Tipo 0
     aBoe[i].nX = (RA.randint(0,nRES[0] - nT_WX) / nT_WX) * nT_WX
     aBoe[i].nY = (RA.randint(0,nRES[1] - nT_HY) / nT_HY) * nT_HY
     aBoe[i].nR = nR_1 - aBoe[i].nX
     aBoe[i].nS = 1 # Switch por defecto
     aBoe[i].dX = 1 # Por defecto robot Direccion Este.-
     aBoe[i].dY = 0
     aBoe[i].nV = 1
     aBoe[i].nC = 1
    return

#---------------------------------------------------------------------
# Inicializa Array de Sprites.-
#---------------------------------------------------------------------
def Init_Fig():
    aImg = []
    aImg.append(Load_Image('T01.png',False )) # Tile Tierra, id = 0
    aImg.append(Load_Image('T02.png',False )) # Tile Roca,   id = 1
    aImg.append(Load_Image('T03.png',False )) # Tile Marmol, id = 2
    aImg.append(Load_Image('Bo1.png',True  )) # Robot 1      id = 3
    aImg.append(Load_Image('Bo2.png',True  )) # Robot 2      id = 4
    aImg.append(Load_Image('Bo3.png',True  )) # Robot 3      id = 5
    aImg.append(Load_Image('Bo4.png',True  )) # Robot 4      id = 6
    aImg.append(Load_Image('Bo5.png',True  )) # Robot 5      id = 7
    aImg.append(Load_Image('Bo6.png',True  )) # Robot 6      id = 8
    aImg.append(Load_Image('Bo7.png',True  )) # Robot 7      id = 9
    aImg.append(Load_Image('Bo8.png',True  )) # Robot 8      id = 10
    aImg.append(Load_Image('Rat.png',True  )) # Mouse 9      id = 11
    aImg.append(Load_Image('T04.png',False )) # Tile OK      id = 12
    aImg.append(Load_Image('T05.png',False )) # Tile Acero   id = 13
    aImg.append(Load_Image('T06.png',False )) # Tile Cobre   id = 14
    aImg.append(Load_Image('T07.png',False )) # Tile Litio   id = 15
    aImg.append(Load_Image('T08.png',False )) # Tile Gas     id = 16
    aImg.append(Load_Image('T09.png',False )) # Tile ?       id = 17
    return aImg

#------------------------------------------------------------------------------
# Armando el Mapa con sus recursos y todos sus parametros para ser utilizado
# por los sensores de los Robots.-
#------------------------------------------------------------------------------
def Init_Mapa():
    for nF in range(0,nRES[1] / nT_HY):
     for nC in range(0,nRES[0] / nT_WX):
      aMap[nF][nC].nT = RA.randint(0,4)  # 0: Baldosa sin Recursos
                                         # 1: Baldosa con Acero
                                         # 2: Baldosa con Cobre
                                         # 3: Baldosa con Litio
                                         # 4: Baldosa con Gas Butano
      aMap[nF][nC].nD = 1  # 1: Disponible - 0: No Disponible
      aMap[nF][nC].nS = 0 # No se pinta por Defecto
      aMap[nF][nC].nF = nF # Fila de la Celda
      aMap[nF][nC].nC = nC # Colu de la Celda
      aMap[nF][nC].nR = aMap[nF][nC].nT
      aMap[nF][nC].nQ = RA.randint(100,1000) # Unidades de RR
    return

#---------------------------------------------------------------------
# Pinta Mapa
#---------------------------------------------------------------------
def Pinta_Mapa():
    for nF in range(0,nRES[1] / nT_HY):
     for nC in range(0,nRES[0] / nT_WX):
      if aMap[nF][nC].nT == 0: # Baldosa sin recursos
         sWin.blit(aFig[02],(aMap[nF][nC].nC*nT_HY,aMap[nF][nC].nF*nT_WX)) # Baldosa sin RR

      if aMap[nF][nC].nT == 1: # Baldosa con Recurso -> Acero
         if aMap[nF][nC].nS == 1: sWin.blit(aFig[13],(aMap[nF][nC].nC*nT_HY,aMap[nF][nC].nF*nT_WX)) # Baldosa con Acero
         else: sWin.blit(aFig[02],(aMap[nF][nC].nC*nT_HY,aMap[nF][nC].nF*nT_WX)) # Baldosa sin RR

      if aMap[nF][nC].nT == 2:  # Baldosa con Recurso -> Cobre
         if aMap[nF][nC].nS == 1: sWin.blit(aFig[14],(aMap[nF][nC].nC*nT_HY,aMap[nF][nC].nF*nT_WX)) # Baldosa con Cobre
         else: sWin.blit(aFig[02],(aMap[nF][nC].nC*nT_HY,aMap[nF][nC].nF*nT_WX)) # Baldosa sin RR

      if aMap[nF][nC].nT == 3:  # Baldosa con Recurso -> Litio
         if aMap[nF][nC].nS == 1: sWin.blit(aFig[15],(aMap[nF][nC].nC*nT_HY,aMap[nF][nC].nF*nT_WX)) # Baldosa con Litio
         else: sWin.blit(aFig[02],(aMap[nF][nC].nC*nT_HY,aMap[nF][nC].nF*nT_WX)) # Baldosa sin RR

      if aMap[nF][nC].nT == 4:  # Baldosa con Recurso -> Gas Butano
         if aMap[nF][nC].nS == 1: sWin.blit(aFig[16],(aMap[nF][nC].nC*nT_HY,aMap[nF][nC].nF*nT_WX)) # Baldosa con Gas Butano
         else: sWin.blit(aFig[02],(aMap[nF][nC].nC*nT_HY,aMap[nF][nC].nF*nT_WX)) # Baldosa sin RR

    return

#---------------------------------------------------------------------
# Modi Mapa
#---------------------------------------------------------------------
def Modi_Mapa(nTile,nOk):
    for nF in range(0,nRES[1] / nT_HY):
     for nC in range(0,nRES[0] / nT_WX):
      if aMap[nF][nC].nT == nTile:
         aMap[nF][nC].nS = nOk # Ahora este TILE se Pinta
    return
#---------------------------------------------------------------------
# Clear Mapa
#---------------------------------------------------------------------
def Clear_Mapa(nVal):
    for nF in range(0,nRES[1] / nT_HY):
     for nC in range(0,nRES[0] / nT_WX):
      aMap[nF][nC].nS = nVal # Limpiamos con CERO
    return                   # Activamos con UNO

#---------------------------------------------------------------------
# Pinta los Robots en el Super Extra Mega Mapa.-
# Se pintan los Robots en Surface -> sMapa (6400 x 480)
#---------------------------------------------------------------------
def Pinta_Robot():
    for i in range(0,nMAX_ROBOTS): # Iteramos las 8 Figuras del Robot
     if aBoe[i].nF == 1: sWin.blit(aFig[3] ,(aBoe[i].nX,aBoe[i].nY))
     if aBoe[i].nF == 2: sWin.blit(aFig[4] ,(aBoe[i].nX,aBoe[i].nY))
     if aBoe[i].nF == 3: sWin.blit(aFig[5] ,(aBoe[i].nX,aBoe[i].nY))
     if aBoe[i].nF == 4: sWin.blit(aFig[6] ,(aBoe[i].nX,aBoe[i].nY))
     if aBoe[i].nF == 5: sWin.blit(aFig[7] ,(aBoe[i].nX,aBoe[i].nY))
     if aBoe[i].nF == 6: sWin.blit(aFig[8] ,(aBoe[i].nX,aBoe[i].nY))
     if aBoe[i].nF == 7: sWin.blit(aFig[9] ,(aBoe[i].nX,aBoe[i].nY))
     if aBoe[i].nF == 8: sWin.blit(aFig[10],(aBoe[i].nX,aBoe[i].nY))
    return

#---------------------------------------------------------------------
# Actualiza la estructura de datos de cada uno de los robots dentro del
# Mapa sMapa.
#---------------------------------------------------------------------
def Mueve_Robot():
    for i in range(0,nMAX_ROBOTS): # Recorrimos todos los Robots
     aBoe[i].nR -= 1      # Decrementamos en 1 el Rango del Robot
     if aBoe[i].nR <= 0:   # Robot termino sus pasos?
        if aBoe[i].nS == 1:
           aBoe[i].nS = 2  # Cambio de estado
           aBoe[i].nR = nR_2 # Robot baja nR_2 pasos
           aBoe[i].dX = 0 ; aBoe[i].dY = 1
        elif aBoe[i].nS == 2:
             aBoe[i].nS = 3  # Cambio de estado
             aBoe[i].nR = nR_1 # Robot OEste nR_1 pasos
             aBoe[i].dX = -1 ; aBoe[i].dY = 0
        elif aBoe[i].nS == 3:
             aBoe[i].nS = 4  # Cambio de estado
             aBoe[i].nR = nR_2 # Robot baja nR_2 pasos
             aBoe[i].dX = 0 ; aBoe[i].dY = 1
        else:
             aBoe[i].nS = 1  # Cambio de estado
             aBoe[i].nR = nR_1 # Robot Este nR_1 pasos
             aBoe[i].dX = 1 ; aBoe[i].dY = 0
     #Actualizamos (Xs,Ys) de los Sprites en el Mapa 2D
     #--------------------------------------------------
     aBoe[i].nX += aBoe[i].dX*aBoe[i].nV # Posicion Robot[i] en eje X
     aBoe[i].nY += aBoe[i].dY*aBoe[i].nV # Posicion Robot[i] en eje Y
     aBoe[i].nC += 1
     if aBoe[i].nC >= 20:
        aBoe[i].nC = 1
        aBoe[i].nF += 1
        if aBoe[i].nF == 9:
           aBoe[i].nF = 1
     if aBoe[i].nX < 1 and aBoe[i].nY == 544: Init_Robot()
    return

#---------------------------------------------------------------------
# Pinta Mouse
#---------------------------------------------------------------------
def Pinta_Mouse():
    sWin.blit(aFig[11],(nMx,nMy))
    return

#--------------------------------------------------------------
# Handle de Pause.-
#--------------------------------------------------------------
def Pausa():
    while 1:
     e = PG.event.wait()
     if e.type in (PG.QUIT, PG.KEYDOWN):
        return

#--------------------------------------------------------------
# SaveData -> Graba todos los datos a un archivo binario
#--------------------------------------------------------------
def SaveData():
    nFh = open('mapa.dat','wb') # Abrimos archivo Mode Binario Escritura
    for nF in range(0,nRES[1] / nT_HY):
     for nC in range(0,nRES[0] / nT_WX):
         eReg = aMap[nF][nC] # Asignamos toda la Info Celda Mapa -> eReg
         nFh.write(eReg)     # Salvamos el registro completo al File
    nFh.close() # Cerramos y vaciamos el buffer RAM File
    return

#---------------------------------------------------------------------
# While Principal del Demo.-
#---------------------------------------------------------------------
aMap = [
        [eCelda() for nC in range(nRES[0]/nT_WX)] for nF in range(nRES[1]/nT_HY)
       ]

sWin = Init_PyGame() ; aFig = Init_Fig()
aBoe = [ eRobot() for i in range(0,nMAX_ROBOTS) ] ; Init_Robot(); Init_Mapa()
aClk = [PG.time.Clock(),PG.time.Clock()] ; eReg = eCelda() ; SaveData()

while lGo:
 cKey = PG.key.get_pressed()
 if cKey[PG.K_ESCAPE] : lGo = ('A' > 'B')

 if cKey[PG.K_p]  : Pausa()
 if cKey[PG.K_s]  : PG.image.save(sWin,'mapinte.png')
 if cKey[PG.K_m]  : Show_Mapa()

 if cKey[PG.K_F1] : Modi_Mapa(1,1) # Permitimos que se pinte Tile 1 Acero
 if cKey[PG.K_F2] : Modi_Mapa(2,1) # Permitimos que se pinte Tile 2 Cobre
 if cKey[PG.K_F3] : Modi_Mapa(3,1) # Permitimos que se pinte Tile 3 Litio
 if cKey[PG.K_F4] : Modi_Mapa(4,1) # Permitimos que se pinte Tile 4 Gas
 if cKey[PG.K_F5] : Clear_Mapa(1)  # Activamos Mapa
 if cKey[PG.K_F6] : Clear_Mapa(0)  # Limpiapos Mapa
 if cKey[PG.K_F7] : SaveData()     # Salvamos el mapa entero a mapa.dat

 ev = PG.event.get()
 for e in ev:
  if e.type == QUIT           : lGo = (2 > 3)
  if e.type == PG.MOUSEMOTION : nMx,nMy = e.pos

 Pinta_Mapa()
 Pinta_Robot()
 Mueve_Robot()
 Pinta_Mouse()
 PG.display.flip()
 aClk[0].tick(100)

PG.quit


