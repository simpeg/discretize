
if __name__ == '__main__':


    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    def topo(x):
        return np.sin(x*(2.*np.pi))*0.3 + 0.5

    def function(cell):
        r = cell.center - np.array([0.5]*len(cell.center))
        dist = np.sqrt(r.dot(r))
        # dist2 = np.abs(cell.center[-1] - topo(cell.center[0]))

        # dist = min([dist1,dist2])
        # if dist < 0.05:
        #     return 5
        if dist < 0.1:
            return 5
        if dist < 0.2:
            return 4
        if dist < 0.4:
            return 3
        return 2

    # T = TreeMesh([[(1,128)],[(1,128)],[(1,128)]],levels=7)
    # T = TreeMesh([128,128,128])
    # T = TreeMesh([64,64],levels=6)
    T = TreeMesh([8,8])
    # T = TreeMesh([[(1,128)],[(1,128)]],levels=7)
    # T.refine(lambda xc:2, balance=False)
    # T._index([0,0,0])
    # T._pointer(0)


    # tic = time.time()
    T.refine(function)#, balance=False)
    # print time.time() - tic
    # print T.nC
    # T.plotSlice(np.log(T.vol))#np.random.rand(T.nC))
    T.plotGrid()
    # print [c for c in T]
    c = T[0]
    plt.plot(c.center[0],c.center[1],'r.')
    nodes = c.nodes
    for n in nodes:
        _ = T._gridN[n,:]
        plt.plot(_[0],_[1],'gs')
    plt.show()
    blah

    # T.plotImage(np.arange(len(T.vol)),showIt=True)

    # print T.getFaceInnerProduct()
    # print T.gridFz


    # T._refineCell([8,0,1])
    # T._refineCell([8,0,2])
    # T._refineCell([12,0,2])
    # T._refineCell([8,4,2])
    # T._refineCell([6,0,3])
    # T._refineCell([8,8,1])
    # T._refineCell([0,0,0,1])
    # T.__dirty__ = True


    # print T.gridFx.shape[0], T.nFx



    ax = plt.subplot(211)
    ax.spy(T.edgeCurl)

    # print Mesh.TensorMesh([2,2,2]).edgeCurl.todense()
    # print T.edgeCurl.todense()
    # print Mesh.TensorMesh([2,2,2]).edgeCurl.todense() - T.edgeCurl.todense()
    # print T.gridEy - Mesh.TensorMesh([2,2,2]).gridEy

    # print T.edge
    # T.plotGrid(ax=ax)

    # R = deflationMatrix(T._facesX, T._hangingFx, T._fx2i)
    # print R

    ax = plt.subplot(212)#, projection='3d')
    ax.spy(Mesh.TensorMesh([2,2,2]).edgeCurl)

    # ax = plt.subplot(313)
    # ax.spy(T.faceDiv[:,:T.nFx] * R)


    # T.balance()
    # T.plotGrid(ax=ax)

    # cx = T._getNextCell([0,0,1],direction=0,positive=True)
    # print cx
    # # print [T._asPointer(_) for _ in cx]
    # cx = T._getNextCell([8,0,3],direction=0,positive=False)
    # print T._asPointer(cx)
    # cx = T._getNextCell([8,8,1],direction=1,positive=False)
    # print cx, #[T._asPointer(_) for _ in cx]
    # cm = T._getNextCell([64,80,4],direction=0,positive=False)
    # cy = T._getNextCell([64,80,4],direction=1,positive=True)
    # cp = T._getNextCell([64,80,4],direction=1,positive=False)

    # ax.plot( T._cellN([4,0,1])[0],T._cellN([4,0,1])[1], 'yd')
    # ax.plot( T._cellN(cx)[0],T._cellN(cx)[1], 'ys')
    # ax.plot( T._cellN(cm)[0],T._cellN(cm)[1], 'ys')
    # ax.plot( T._cellN(cy)[0],T._cellN(cy)[1], 'ys')
    # ax.plot( T._cellN(cp[0])[0],T._cellN(cp[0])[1], 'ys')
    # ax.plot( T._cellN(cp[1])[0],T._cellN(cp[1])[1], 'ys')





    # print T.nN

    plt.show()

