import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser

from embedding_net import EmbeddingNet
from deepgesturedataset import DeepGestureDataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_iter(target_data, net, optimizer, device):
    # zero gradients
    optimizer.zero_grad()

    # reconstruction loss
    net.to(device)
    feat, recon_data = net(target_data)
    recon_loss = F.l1_loss(recon_data, target_data, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))

    if True:  # use pose diff
        target_diff = target_data[:, 1:] - target_data[:, :-1]
        recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
        recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

    recon_loss = torch.sum(recon_loss)

    recon_loss.backward()
    optimizer.step()

    ret_dict = {'loss': recon_loss.item()}
    return ret_dict

#
# def make_tensor(path, n_frames, stride=None):
#     # use calculate_mean_std.py to get mean/std values
#     if 'fullbody' in path:
#         mean_vec = np.array(
#             [0.01855118070268579, -0.1013322468846258, 0.08283318522940754, 0.01855118070268579, 182.54484156276828, 0.08283318522940754, 0.3541669587027638, 186.68091329162775, -6.311844557005045, 1.314324011474814, 208.66832174324736, -4.261926167518727, 2.080280203610869, 229.51662355065372, -12.274310672197933, 2.747281942064785, 266.0706399098627, -16.4480033015734, 3.1459839022120244, 291.5177753558705, -10.492305344072888, 3.436658179697915, 303.9450667015395, -4.166701483381484,
#              -2.157822697191497, 280.6766966372547, 1.3016120858482163, -33.915650414979325, 278.3425489365996, -7.857284363684173, -34.75230898936223, 270.16624429917937, -8.331751686015162, -34.75230898936223, 270.16624429917937, -8.331751686015162, -43.668024861185906, 221.27271982813915, -0.1946196412603046, -50.7421176330811, 191.19445966969863, 24.00335486713785, -48.610832074672665, 189.37390388270728, 27.078022509963724, -47.29041767053703, 179.8943203750877, 36.8866407849576,
#              -48.112188162577944, 173.76535301149983, 38.03657590004193, -46.95958063987163, 170.846477306871, 36.90238887289396, -53.569113667176865, 179.76709799379563, 34.03215922679267, -53.44144541233228, 173.3768409377897, 33.2551234065568, -51.62412431579907, 170.0636317076318, 31.314338860028595, -50.96893111644582, 179.92625777511927, 36.13865433055161, -51.13066680801513, 172.77549730500658, 35.42082909407342, -48.86708324704132, 169.74849040715324, 33.2720614863769,
#              -55.512876227685375, 179.43790411890703, 31.39773750037596, -57.34440757776709, 174.77451931256542, 32.43167126863285, -56.60496667361927, 171.9564091422062, 31.477530364495383, -47.44861215834778, 188.15637666520232, 28.12902459703336, -43.55625468665576, 185.47388611507412, 27.758374254086043, -41.381992678513654, 180.42104207281443, 30.551513582140757, -41.691166629453335, 175.90447296043635, 33.04638334707732, 8.854384931033676, 280.5783099180194, 0.9996880195461859,
#              40.20013135374428, 278.44674967572644, -9.592326671467356, 40.89415682483461, 270.1516950586752, -10.15633373951769, 40.89415682483461, 270.1516950586752, -10.15633373951769, 51.41500406841752, 221.70370614542432, -2.6006996477813753, 54.916744401223745, 194.5284763365179, 24.311837550207308, 52.31299778157673, 192.81266112279957, 26.82634961566674, 51.00542364029097, 191.68991603881014, 27.502150006608233, 47.555715817790315, 189.24028792058152, 26.23880539910951,
#              45.18781599616401, 184.54008965581176, 28.23322574269807, 44.39069379886114, 180.4513845313832, 30.530440578637357, 49.10887545792274, 184.0275356958452, 35.706450470870244, 49.90993964505175, 178.3885187020955, 36.84489606195685, 49.30781264276034, 175.7841518499736, 35.57328214365982, 52.82784436525587, 184.04666654777262, 36.08042940358183, 52.967532008378505, 177.7675448820657, 35.25556966590845, 51.657114166295656, 175.27654226196464, 32.718515505034006, 55.88423735307892,
#              183.90380449262895, 34.959872588070354, 56.03003553094789, 178.02777481146896, 34.671341338435475, 55.09303635629754, 174.87312861774282, 32.75722625861554, 58.46223556189697, 183.61085660188184, 33.19737390418257, 59.8327591426162, 179.22367432704505, 34.806910217356176, 59.45952664499003, 176.66793963961754, 33.844020088240285, -17.91396729720144, 169.09892066561477, -0.8529426350725592, -20.49614301073527, 96.3864404375965, 7.23718680507571, 16.321097420727913,
#              167.11616457531628, -0.2859152140551241, 23.352464277737283, 95.1412407184394, 10.641219949816891])
#         std_vec = np.array(
#             [9.719021721795063, 0.5660665457076898, 4.831627219692865, 9.719021721795063, 2.10084851965359, 4.831627219692865, 9.350322073106716, 2.27767028126296, 4.862355228679545, 8.409351971104883, 2.2284892262765488, 5.192218382679325, 8.2655019967835, 2.305365980499011, 5.411700511947982, 9.750621346247549, 2.350968263959134, 6.4917375754044775, 11.893218925406007, 2.3373181572534993, 7.706785781159235, 11.73201991166551, 2.552488775724235, 7.8370339127014255, 11.384635161537684,
#              2.513133360704533, 7.178278048030584, 11.263408651888561, 4.2417286038903805, 9.648342762259025, 10.764373855438103, 4.356302017640735, 9.296473674862655, 10.764373855438103, 4.356302017640735, 9.296473674862655, 10.832343970345711, 6.340108061608368, 10.983591368195322, 14.066483666563256, 29.138061955942977, 15.816309303360223, 14.674993695152681, 32.021790106841266, 15.634180468843526, 17.8940953042104, 42.49489566237325, 17.366167167123358, 19.283225572738022, 46.72490660155962,
#              19.92092457491629, 19.846855750918, 48.37379116501593, 21.73504842450431, 17.66380549386174, 39.6404559070494, 19.154592476436427, 18.890404901432888, 42.88183271752636, 22.17803722730876, 19.355251492783697, 44.341441869234956, 24.185137312723214, 17.872431231111243, 41.26342268246586, 18.163717848051974, 19.167379973772654, 44.994565871258786, 21.54399335823723, 19.502396643627215, 46.30957662481494, 23.568680667437945, 17.49421774720494, 37.978752158303436, 20.276846451872977,
#              18.887165734196895, 40.609422882552266, 22.455498253623045, 19.42860181623181, 41.841209135644974, 23.85796378483092, 14.969283669309124, 33.45111101883678, 15.64759076893061, 15.114846518629307, 35.462455580218645, 16.00922836924407, 16.34775303526974, 40.7792680666506, 17.14058661058859, 17.8131799043824, 45.0966553239018, 18.6445204775486, 11.303156158017574, 2.559557656325625, 7.068536721992277, 11.014860596790749, 3.820297854783159, 9.27840262887583, 10.608198525051565,
#              3.859980561092301, 9.053027560599258, 10.608198525051565, 3.859980561092301, 9.053027560599258, 10.161361789846968, 5.807387131576938, 11.485124437716644, 15.902184560343224, 28.747362645142367, 17.71892333015811, 16.318607140818735, 31.819301497311628, 17.880586508667523, 16.557950813189237, 33.36606497709961, 18.114144141111865, 16.825982748414837, 35.54599400401188, 19.154006180475445, 18.33841248133445, 41.1129648463966, 21.079555467725996, 19.94257707436061, 45.85876872948868,
#              22.91105288499688, 19.56636278479743, 42.97220536569384, 20.470437737577885, 21.640400737877904, 47.16842379955284, 23.790588346195467, 22.637429082734254, 48.68878954269123, 26.20803108076218, 19.920787193620804, 41.548175037033786, 20.841692202715482, 22.063248631210147, 45.03942685191612, 24.913442247326014, 22.947406830608514, 46.08461726633099, 27.414753034487514, 20.18639657648188, 39.70737765578248, 21.52422530692076, 22.214070591184935, 42.98622692299402, 24.96486414244103,
#              23.33861954365486, 44.472346774193284, 27.384960746263754, 20.531833818380665, 37.83148312448459, 22.410933213366178, 22.34585045660999, 40.61023963879408, 24.645464268816095, 23.265280956264295, 41.789764891152494, 26.25772635359072, 10.859272953514816, 2.9156970504137303, 5.628154906410027, 7.240719441411309, 3.117290232989621, 11.918614434923379, 10.990471030928338, 2.7193944341376555, 5.180799178578772, 5.590297080069878, 2.9619588368791585, 10.530597551623426])
#     else:
#         mean_vec = np.array(
#             [0.01855118070268579, -0.1013322468846258, 0.08283318522940754, 0.01855118070268579, 182.54484156276828, 0.08283318522940754, 0.35416695870276377, 186.68091329162775, -6.3118445570050445, 1.3143240114748138, 208.66832174324736, -4.261926167518727, 2.080280203610869, 229.51662355065372, -12.274310672197933, 2.747281942064785, 266.0706399098627, -16.448003301573394, 3.1459839022120244, 291.5177753558705, -10.492305344072886, 3.4366581796979148, 303.9450667015394, -4.166701483381484,
#              -2.157822697191497, 280.6766966372547, 1.3016120858482165, -33.915650414979325, 278.3425489365996, -7.857284363684172, -34.75230898936223, 270.16624429917937, -8.33175168601516, -34.75230898936223, 270.16624429917937, -8.33175168601516, -43.668024861185906, 221.27271982813915, -0.1946196412603046, -50.742117633081094, 191.19445966969863, 24.00335486713785, -48.61083207467266, 189.37390388270728, 27.078022509963724, -47.290417670537025, 179.8943203750877, 36.8866407849576,
#              -48.11218816257794, 173.76535301149983, 38.03657590004193, -46.95958063987163, 170.846477306871, 36.90238887289396, -53.569113667176865, 179.76709799379563, 34.03215922679267, -53.44144541233228, 173.3768409377897, 33.2551234065568, -51.62412431579906, 170.0636317076318, 31.314338860028588, -50.96893111644582, 179.92625777511927, 36.13865433055161, -51.13066680801512, 172.77549730500658, 35.42082909407341, -48.867083247041315, 169.74849040715324, 33.2720614863769,
#              -55.51287622768536, 179.43790411890703, 31.39773750037596, -57.34440757776708, 174.77451931256542, 32.43167126863285, -56.604966673619266, 171.9564091422062, 31.47753036449538, -47.44861215834778, 188.15637666520232, 28.129024597033354, -43.55625468665575, 185.47388611507412, 27.758374254086043, -41.38199267851365, 180.42104207281443, 30.551513582140757, -41.691166629453335, 175.90447296043635, 33.04638334707732, 8.854384931033676, 280.57830991801933, 0.9996880195461864,
#              40.20013135374427, 278.44674967572644, -9.592326671467355, 40.89415682483461, 270.1516950586752, -10.15633373951769, 40.89415682483461, 270.1516950586752, -10.15633373951769, 51.41500406841752, 221.70370614542432, -2.6006996477813753, 54.916744401223745, 194.5284763365179, 24.311837550207308, 52.31299778157673, 192.81266112279957, 26.82634961566674, 51.005423640290964, 191.68991603881014, 27.502150006608233, 47.555715817790315, 189.24028792058152, 26.238805399109506,
#              45.18781599616401, 184.54008965581176, 28.233225742698068, 44.39069379886114, 180.45138453138327, 30.530440578637357, 49.108875457922736, 184.02753569584522, 35.70645047087024, 49.90993964505174, 178.3885187020955, 36.84489606195685, 49.307812642760325, 175.7841518499736, 35.57328214365982, 52.82784436525587, 184.04666654777262, 36.080429403581824, 52.967532008378505, 177.7675448820657, 35.25556966590845, 51.65711416629565, 175.27654226196464, 32.718515505034006, 55.884237353078916,
#              183.90380449262895, 34.959872588070354, 56.030035530947885, 178.02777481146896, 34.671341338435475, 55.09303635629753, 174.87312861774282, 32.75722625861553, 58.46223556189697, 183.61085660188184, 33.19737390418257, 59.8327591426162, 179.22367432704505, 34.806910217356176, 59.45952664499003, 176.66793963961754, 33.844020088240285])
#         std_vec = np.array(
#             [9.719021721795063, 0.5660665457076898, 4.831627219692865, 9.719021721795063, 2.10084851965359, 4.831627219692865, 9.350322073106716, 2.27767028126296, 4.862355228679545, 8.409351971104885, 2.2284892262765483, 5.192218382679325, 8.2655019967835, 2.305365980499011, 5.411700511947982, 9.750621346247549, 2.3509682639591336, 6.491737575404477, 11.893218925406007, 2.3373181572534993, 7.706785781159234, 11.73201991166551, 2.552488775724235, 7.837033912701425, 11.384635161537684,
#              2.513133360704533, 7.178278048030583, 11.263408651888561, 4.24172860389038, 9.648342762259023, 10.764373855438105, 4.356302017640735, 9.296473674862655, 10.764373855438105, 4.356302017640735, 9.296473674862655, 10.832343970345711, 6.340108061608367, 10.98359136819532, 14.066483666563256, 29.138061955942977, 15.816309303360221, 14.674993695152681, 32.021790106841266, 15.634180468843525, 17.894095304210396, 42.49489566237325, 17.366167167123358, 19.28322557273802, 46.72490660155961,
#              19.920924574916288, 19.846855750918, 48.37379116501592, 21.735048424504306, 17.66380549386174, 39.64045590704939, 19.154592476436427, 18.890404901432888, 42.88183271752634, 22.17803722730876, 19.355251492783697, 44.341441869234956, 24.185137312723214, 17.87243123111124, 41.26342268246586, 18.163717848051974, 19.167379973772654, 44.99456587125878, 21.54399335823723, 19.502396643627215, 46.30957662481492, 23.568680667437945, 17.49421774720494, 37.978752158303436, 20.276846451872977,
#              18.88716573419689, 40.609422882552266, 22.45549825362304, 19.42860181623181, 41.841209135644974, 23.857963784830915, 14.969283669309124, 33.45111101883677, 15.64759076893061, 15.114846518629307, 35.46245558021864, 16.00922836924407, 16.34775303526974, 40.7792680666506, 17.14058661058859, 17.813179904382398, 45.09665532390179, 18.6445204775486, 11.303156158017574, 2.5595576563256244, 7.068536721992276, 11.014860596790749, 3.820297854783159, 9.27840262887583, 10.608198525051565,
#              3.8599805610923004, 9.053027560599258, 10.608198525051565, 3.8599805610923004, 9.053027560599258, 10.161361789846968, 5.807387131576938, 11.485124437716644, 15.902184560343224, 28.747362645142363, 17.71892333015811, 16.31860714081873, 31.819301497311628, 17.880586508667523, 16.557950813189237, 33.36606497709961, 18.114144141111865, 16.825982748414837, 35.54599400401188, 19.154006180475445, 18.33841248133445, 41.11296484639659, 21.079555467725992, 19.94257707436061,
#              45.85876872948868, 22.91105288499688, 19.56636278479743, 42.97220536569383, 20.470437737577882, 21.640400737877904, 47.16842379955284, 23.790588346195467, 22.637429082734254, 48.68878954269122, 26.208031080762176, 19.9207871936208, 41.54817503703378, 20.841692202715482, 22.06324863121015, 45.03942685191612, 24.91344224732601, 22.947406830608514, 46.08461726633098, 27.41475303448751, 20.18639657648188, 39.70737765578248, 21.52422530692076, 22.214070591184935, 42.986226922994014,
#              24.96486414244103, 23.33861954365486, 44.47234677419327, 27.384960746263754, 20.531833818380665, 37.831483124484585, 22.410933213366174, 22.345850456609988, 40.61023963879408, 24.645464268816095, 23.26528095626429, 41.78976489115249, 26.257726353590716])
#
#     if os.path.isdir(path):
#         files = glob.glob(os.path.join(path, '*.npy'))
#     else:
#         files = [path]
#
#     samples = []
#     stride = n_frames // 2 if stride is None else stride
#     for file in files:
#         data = np.load(file)
#         for i in range(0, len(data) - n_frames, stride):
#             sample = data[i:i + n_frames]
#             sample = (sample - mean_vec) / std_vec
#             samples.append(sample)
#
#     return torch.Tensor(np.array(samples))


def main(args, gesture_dim, n_frames, device):
    dataset = DeepGestureDataset(dataset_file=args.dataset)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # train
    loss_meters = [AverageMeter('loss')]

    # interval params
    print_interval = int(len(train_loader) / 5)

    model = EmbeddingNet(gesture_dim, n_frames).to(device)
    gen_optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.5, 0.999))

    # training
    for epoch in range(args.epoch):
        print('Epoch {}/{}'.format(epoch + 1, args.epoch))
        for i, batch in enumerate(train_loader, 0):
            batch_gesture = np.asarray(batch, np.float32)
            target_vec = torch.Tensor(batch_gesture).to(device)
            loss = train_iter(target_vec, model, gen_optimizer, device)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], args.batch_size)

            # print training status
            if (i + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | '.format(epoch, i + 1)
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                print(print_summary)

    # save model
    state_dict = model.cpu().state_dict()
    file_path = f'output/model_checkpoint_{gesture_dim}_{n_frames}.bin'
    torch.save({'gesture_dim': gesture_dim, 'n_frames': n_frames, 'gen_dict': state_dict}, file_path)


if __name__ == '__main__':
    """
    python train_AE.py --dataset=../data/real_dataset.npz --gpu=cuda:0
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--dataset', '-d', required=True, default="../data/real_dataset.npz",
                        help="")
    parser.add_argument('--gpu', '-gpu', required=True, default="cuda:0",
                        help="")
    parser.add_argument('--epoch', '-epoch', type=int, default=50,
                        help="")
    parser.add_argument('--batch_size', '-bs', type=int, default=64,
                        help="")

    args = parser.parse_args()

    n_frames = 88
    gesture_dim = 1141
    device = torch.device(args.gpu)

    main(args, gesture_dim, n_frames, device)
