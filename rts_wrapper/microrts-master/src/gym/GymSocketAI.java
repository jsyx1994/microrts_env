package gym;

import ai.socket.SocketAI;
import rts.units.UnitTypeTable;

public class GymSocketAI extends SocketAI {
    public GymSocketAI(UnitTypeTable a_utt) {
        super(a_utt);
    }

    public GymSocketAI(int mt, int mi, String a_sa, int a_port, int a_language, UnitTypeTable a_utt) {
        super(mt, mi, a_sa, a_port, a_language, a_utt);
    }
}
